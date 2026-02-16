from __future__ import annotations

import warnings
from typing import Dict, Any, Optional

import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_prominences, peak_widths
import traceback

# ======================================================================
# ============================ MAIN ====================================
# ======================================================================

def sdfind(
    denoised_data: np.ndarray,
    peak: np.ndarray,
    LIZ: np.ndarray,
    CONC: np.ndarray,
    x: Optional[np.ndarray] = None,
    override_repers: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Детекция пиков геномной библиотеки + построение калибровки.

    Parameters
    ----------
    denoised_data : np.ndarray
        Предобработанный сигнал (baseline removed, denoised).

    peak : np.ndarray
        Позиции реперных пиков (индексы).

    LIZ : np.ndarray
        Длины фрагментов стандарта (bp).

    CONC : np.ndarray
        Концентрации стандарта (необязательно используются).

    x : np.ndarray | None, optional
        Ось индексов. Если None — создаётся автоматически.

    override_repers : np.ndarray | None, optional
        Ручные реперы (если заданы — используются вместо авто).

    Returns
    -------
    result : dict
        {
            "selected_peaks": np.ndarray[int],
            "st_length": np.ndarray[int],
            "LibPeakLocations": np.ndarray[int],
            "unrecognized_peaks": np.ndarray[int],
            "t_main": np.ndarray[float],
            "mainCorr": np.ndarray[float],
            "calibration": {
                "x_cal": np.ndarray,
                "y_cal": np.ndarray,
                "poly_coeff": np.ndarray,
            },
            "denoised_data": np.ndarray,
        }

    Notes
    -----
    Pipeline:

    1. Savitzky–Golay smoothing
    2. 2-я производная (MATLAB-style)
    3. MATLAB-подобный findpeaks
    4. Система баллов Points
    5. Автовыбор реперов
    6. Полиномиальная калибровка
    """

    # ================================================================
    # safety & copy
    # ================================================================
    denoised_data = np.asarray(denoised_data, dtype=np.float64).copy()
    denoised_data[~np.isfinite(denoised_data)] = 0
    denoised_data[denoised_data < 0] = 0

    peak = np.asarray(peak, dtype=np.float64)
    LIZ = np.asarray(LIZ, dtype=np.float64)
    CONC = np.asarray(CONC, dtype=np.float64)

    if x is None:
        x = np.arange(len(denoised_data), dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64)

    # ================================================================
    # empty template
    # ================================================================
    def _empty() -> Dict[str, Any]:
        return {
            "selected_peaks": np.array([], dtype=int),
            "st_length": np.array([], dtype=int),
            "LibPeakLocations": np.array([], dtype=int),
            "unrecognized_peaks": np.array([], dtype=int),
            "t_main": np.array([], dtype=float),
            "mainCorr": np.array([], dtype=float),
            "calibration": {
                "x_cal": np.array([], dtype=float),
                "y_cal": np.array([], dtype=float),
                "poly_coeff": np.array([], dtype=float),
            },
            "denoised_data": denoised_data,
        }

    if len(denoised_data) < 20:
        return _empty()

    # ================================================================
    # 1. Savitzky–Golay
    # ================================================================
    try:
        filt = savgol_filter(denoised_data, 5, 1)
        d1 = np.diff(filt)
        filt_d1 = savgol_filter(d1, 5, 1)
        d2 = np.diff(filt_d1)
        filt_d2 = savgol_filter(d2, 5, 1)
    except Exception:
        return _empty()

    flipped = -filt_d2
    flipped[flipped < 0] = 0

    # ================================================================
    # 2. findpeaks MATLAB-style
    # ================================================================
    peaks_v, peak_locs, peak_w, peak_p = findpeaks_matlab(flipped)

    if peak_locs.size == 0:
        return _empty()

    # ================================================================
    # 3. отсев краёв
    # ================================================================
    valid = (peak_locs > 15) & (peak_locs < (len(x) - 10))

    peak_locs = peak_locs[valid]
    peaks_v = peaks_v[valid]
    peak_w = peak_w[valid]
    peak_p = peak_p[valid]

    if peak_locs.size < 2:
        return _empty()

    # ================================================================
    # 4. система баллов Points
    # ================================================================
    Points = np.zeros_like(peak_locs, dtype=np.float64)

    # --- ширины ---
    idx_w = np.argsort(peak_w)[::-1]
    sw = peak_w[idx_w]
    if sw.size > 1 and (sw.max() - sw.min()) > 0:
        norm_w = (sw - sw.min()) / (sw.max() - sw.min())
    else:
        norm_w = np.ones_like(sw)
    Points[idx_w] += norm_w

    # --- отношение ---
    R = peaks_v / np.where(peak_w == 0, 1e-12, peak_w)
    idx_r = np.argsort(R)[::-1]
    sr = R[idx_r]
    if sr.size > 1 and (sr.max() - sr.min()) > 0:
        norm_r = (sr - sr.min()) / (sr.max() - sr.min())
    else:
        norm_r = np.ones_like(sr)
    Points[idx_r] += norm_r

    # ================================================================
    # 5. выбор реперов
    # ================================================================
    if override_repers is not None and len(override_repers) >= 2:
        st_length = np.sort(np.asarray(override_repers, dtype=int))
    else:
        sorted_idx = np.argsort(Points)[::-1]
        st_length = np.sort(peak_locs[sorted_idx[:2]])

    # ================================================================
    # 6. калибровка времени
    # ================================================================
    try:
        px = np.polyfit(st_length, [peak[0], peak[-1]], 1)
        traceback.print_exc()
    except Exception:
        return _empty()

    t = np.arange(len(denoised_data), dtype=np.float64)
    t_main = np.polyval(px, t)

    # ================================================================
    # 7. главная калибровка bp
    # ================================================================
    try:
        SDC = np.polyfit(peak, LIZ, min(5, len(peak) - 1))
        traceback.print_exc()
    except Exception:
        return _empty()

    mainCorr = np.polyval(SDC, t_main)

    # ================================================================
    # 8. библиотечные пики
    # ================================================================
    mask_between = (
        (peak_locs >= st_length[0]) &
        (peak_locs <= st_length[-1])
    )

    selectedPeakLocations = peak_locs[mask_between]

    # ================================================================
    # 9. калибровочная кривая (для PDF!)
    # ================================================================
    x_cal = peak.astype(np.float64)
    y_cal = LIZ.astype(np.float64)

    # ================================================================
    # RESULT
    # ================================================================
    return {
        "selected_peaks": selectedPeakLocations.astype(int),
        "st_length": st_length.astype(int),
        "LibPeakLocations": selectedPeakLocations.astype(int),
        "unrecognized_peaks": np.array([], dtype=int),
        "t_main": t_main,
        "mainCorr": mainCorr,
        "calibration": {
            "x_cal": x_cal,
            "y_cal": y_cal,
            "poly_coeff": SDC,
        },
        "denoised_data": denoised_data,
    }


# ======================================================================
# ===================== MATLAB findpeaks ===============================
# ======================================================================

def findpeaks_matlab(
    data: np.ndarray,
    min_peak_height: Optional[float] = None,
    min_peak_distance: Optional[int] = None,
):
    """
    MATLAB-подобный findpeaks.

    Returns
    -------
    values : np.ndarray
    locations : np.ndarray[int]
    widths : np.ndarray
    prominences : np.ndarray
    """

    data = np.asarray(data, dtype=np.float64)

    kw = {}
    if min_peak_height is not None:
        kw["height"] = float(min_peak_height)
    if min_peak_distance is not None:
        kw["distance"] = int(min_peak_distance)

    locations, _ = find_peaks(data, **kw)

    if locations.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, locations.astype(int), empty, empty

    values = data[locations]
    prominences = peak_prominences(data, locations)[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        widths = peak_widths(data, locations, rel_height=0.5)[0]

    return values, locations.astype(int), widths, prominences

