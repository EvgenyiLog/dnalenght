import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.signal import savgol_filter


def sdfind(
    denoised_data: np.ndarray,
    peak: np.ndarray,
    LIZ: np.ndarray,
    CONC: Optional[np.ndarray]=None,
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
        Концентрации стандарта.

    x : np.ndarray | None
        Ось индексов (если None — создаётся автоматически).

    override_repers : np.ndarray | None
        Ручные реперы.

    Returns
    -------
    result : dict
        {
            "selected_peaks": np.ndarray,
            "st_length": np.ndarray,
            "LibPeakLocations": np.ndarray,
            "unrecognized_peaks": np.ndarray,
            "t_main": np.ndarray,
            "mainCorr": np.ndarray,
            "calibration": {
                "x_cal": np.ndarray,
                "y_cal": np.ndarray,
                "poly_coeff": np.ndarray,
            },
            "denoised_data": np.ndarray,
        }

    Notes
    -----
    Алгоритм:

    1. Savitzky–Golay фильтрация
    2. 2-я производная
    3. findpeaks (MATLAB-style)
    4. Система баллов Points
    5. Автовыбор реперов
    6. Линейная калибровка

    Совместимо с pipeline анализа SCF/геномных библиотек.
    """

    # ================================================================
    # safety
    # ================================================================
    denoised_data = np.asarray(denoised_data, dtype=float).copy()
    denoised_data[denoised_data < 0] = 0

    if x is None:
        x = np.arange(len(denoised_data))

    _empty = {
        "selected_peaks": np.array([], dtype=int),
        "st_length": np.array([], dtype=int),
        "LibPeakLocations": np.array([], dtype=int),
        "unrecognized_peaks": np.array([], dtype=int),
        "t_main": np.array([]),
        "mainCorr": np.array([]),
        "calibration": {
            "x_cal": np.array([]),
            "y_cal": np.array([]),
            "poly_coeff": np.array([]),
        },
        "denoised_data": denoised_data,
    }

    # ================================================================
    # 2. фильтрация и производные
    # ================================================================
    filt = savgol_filter(denoised_data, 5, 1)
    d1 = np.diff(filt)
    filt_d1 = savgol_filter(d1, 5, 1)
    d2 = np.diff(filt_d1)
    filt_d2 = savgol_filter(d2, 5, 1)

    flipped = -filt_d2.copy()
    flipped[flipped < 0] = 0

    # ⚠️ твоя функция должна существовать
    peaks_v, peak_locs, peak_w, peak_p = findpeaks_matlab(flipped)

    if len(peak_locs) == 0:
        return _empty

    # ================================================================
    # отсев краёв
    # ================================================================
    valid = (peak_locs > 15) & (peak_locs < (len(x) - 10))
    peak_locs = peak_locs[valid]
    peaks_v = peaks_v[valid]
    peak_w = peak_w[valid]
    peak_p = peak_p[valid]

    if len(peak_locs) == 0:
        return _empty

    # ================================================================
    # система баллов Points (твоя логика сохранена)
    # ================================================================
    Points = np.zeros(len(peak_p), dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        orders = np.floor(np.log10(np.abs(peak_p)))
    orders[~np.isfinite(orders)] = 0
    max_order_val = np.max(orders)

    # --- ширины ---
    idx_w = np.argsort(peak_w)[::-1]
    sw = peak_w[idx_w]
    if len(sw) > 1:
        rng = sw.max() - sw.min()
        norm_w = (sw - sw.min()) / rng if rng > 0 else np.ones_like(sw)
    else:
        norm_w = np.array([1.0])
    Points[idx_w] += norm_w

    # --- отношение ---
    R = peaks_v / np.where(peak_w == 0, 1e-10, peak_w)
    idx_r = np.argsort(R)[::-1]
    sr = R[idx_r]
    if len(sr) > 1:
        rng = sr.max() - sr.min()
        norm_r = (sr - sr.min()) / rng if rng > 0 else np.ones_like(sr)
    else:
        norm_r = np.array([1.0])
    Points[idx_r] += norm_r

    # ================================================================
    # 🔴 УПРОЩЁННЫЙ выбор реперов (стабильный)
    # ================================================================
    sorted_idx = np.argsort(Points)[::-1]

    if len(sorted_idx) < 2:
        return _empty

    first_reper = peak_locs[sorted_idx[0]]
    second_reper = peak_locs[sorted_idx[1]]

    st_length = np.sort([first_reper, second_reper])

    # ================================================================
    # 4. КАЛИБРОВКА
    # ================================================================
    try:
        px = np.polyfit(st_length, [peak[0], peak[-1]], 1)
    except Exception:
        return _empty

    t = np.arange(len(denoised_data), dtype=float)
    t_main = np.polyval(px, t)

    # 🔥 ГЛАВНАЯ калибровка bp
    SDC = np.polyfit(peak, LIZ, 5)
    mainCorr = np.polyval(SDC, t_main)

    # ================================================================
    # библиотека (минимальная версия)
    # ================================================================
    mask_between = (
        (peak_locs >= st_length[0]) &
        (peak_locs <= st_length[-1])
    )

    selectedPeakLocations = peak_locs[mask_between]

    LibPeakLocations = selectedPeakLocations.copy()
    unrecognized_peaks = np.array([], dtype=int)

    # ================================================================
    # 📈 калибровочная кривая для PDF
    # ================================================================
    x_cal = peak.astype(float)
    y_cal = LIZ.astype(float)

    result = {
        "selected_peaks": selectedPeakLocations.astype(int),
        "st_length": st_length.astype(int),
        "LibPeakLocations": LibPeakLocations.astype(int),
        "unrecognized_peaks": unrecognized_peaks,
        "t_main": t_main,
        "mainCorr": mainCorr,
        "calibration": {
            "x_cal": x_cal,
            "y_cal": y_cal,
            "poly_coeff": SDC,
        },
        "denoised_data": denoised_data,
    }

    return result
