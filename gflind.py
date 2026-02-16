from __future__ import annotations

import warnings
import numpy as np
from typing import Tuple, Optional

from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import gaussian_filter1d


def glfind(
    raw_ref: np.ndarray,
    zr_ref: np.ndarray,
    in_liz: np.ndarray,
    conc: np.ndarray,
    locs_pol: Optional [np.ndarray] = None
    
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Поиск пиков стандарта длин (LIZ) в сигнале — аналог glfind (MATLAB-style).

    Функция:
    - разворачивает сигнал (MATLAB flipud)
    - аппроксимирует ожидаемые позиции пиков
    - выполняет адаптивный авто-поиск пиков
    - вычисляет площади пиков
    - возвращает молярности

    Параметры
    ----------
    raw_ref : np.ndarray
        Исходный сигнал (интенсивность).
    zr_ref : np.ndarray
        Обработанный сигнал для поиска пиков.
    in_liz : np.ndarray
        Эталонные длины LIZ.
    locs_pol : np.ndarray
        Ожидаемые позиции пиков (может быть нулевым).
    conc : np.ndarray
        Концентрации стандартов.

    Возвращает
    ----------
    locs2 : np.ndarray[int]
        Позиции найденных пиков.
    area : np.ndarray[float]
        Площади пиков.
    raw_ref_out : np.ndarray[float]
        Сигнал после обратного flip (MATLAB-совместимость).
    sd_molarity : np.ndarray[float]
        Рассчитанные молярности.

    Примечания
    ----------
    - Без baseline drift
    - Без wden
    - MATLAB-совместимая логика flip
    - Устойчив к шуму
    """

    # ============================================================
    # 1. Приведение типов
    # ============================================================
    raw_ref = np.asarray(raw_ref, dtype=np.float64).ravel()
    zr_ref = np.asarray(zr_ref, dtype=np.float64).ravel()
    in_liz = np.asarray(in_liz, dtype=np.float64).ravel()
    locs_pol = np.asarray(locs_pol, dtype=np.float64).ravel()
    conc = np.asarray(conc, dtype=np.float64).ravel()

    if raw_ref.size == 0 or zr_ref.size == 0 or in_liz.size == 0:
        return _empty_result(raw_ref)

    # ============================================================
    # 2. Разворот LIZ (MATLAB логика)
    # ============================================================
    LIZ = in_liz[::-1]
    n_liz = len(LIZ)

    # ============================================================
    # 3. Flip сигналов (как MATLAB flipud)
    # ============================================================
    zr_ref = zr_ref[::-1].copy()
    raw_ref = raw_ref[::-1].copy()

    # ============================================================
    # 4. Аппроксимация ожидаемых позиций
    # ============================================================
    if np.all(locs_pol == 0) or len(locs_pol) == 0:
        n_pts = len(zr_ref)
        locs_pol = np.linspace(n_pts * 0.1, n_pts * 0.9, n_liz)

    poly_order = min(4, n_liz - 1)
    z = np.polyfit(in_liz, locs_pol, poly_order)
    new_LIZ = np.polyval(z, LIZ)
    dLIZ = np.abs(np.diff(new_LIZ))

    if len(dLIZ) == 0 or dLIZ[0] == 0:
        return _empty_result(raw_ref)

    # ============================================================
    # 5. Автоматический поиск пиков
    # ============================================================
    locs2 = _autofind_peaks(zr_ref, LIZ, dLIZ)

    if locs2 is None or len(locs2) == 0:
        return _empty_result(raw_ref)

    # ============================================================
    # 6. Расчёт площадей
    # ============================================================
    area = np.array(
        [
            np.trapezoid(raw_ref[max(0, i - 3): i + 4])
            for i in locs2
        ],
        dtype=np.float64,
    )

    # ============================================================
    # 7. Обратный flip (MATLAB совместимость)
    # ============================================================
    n_pts = len(raw_ref)
    raw_ref_out = raw_ref[::-1]
    locs2 = (n_pts - 1 - locs2)[::-1]

    # ============================================================
    # 8. Расчёт молярности
    # ============================================================
    sd_molarity = ((conc * 1e-3) / (649.0 * in_liz)) * 1e9

    return locs2.astype(int), area, raw_ref_out.astype(float), sd_molarity.astype(float)


# ======================================================================
# ========================= ВСПОМОГАТЕЛЬНЫЕ ===========================
# ======================================================================

def _autofind_peaks(
    data: np.ndarray,
    LIZ: np.ndarray,
    dLIZ: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Устойчивый авто-поиск пиков.
    """

    data = np.asarray(data, dtype=np.float64).ravel()
    n_liz = len(LIZ)

    if n_liz < 2 or len(data) < 20:
        return None

    # мягкое сглаживание (без wden!)
    data_smooth = gaussian_filter1d(data.copy(), sigma=1.5)
    data_smooth[data_smooth < 0] = 0

    best_result = None
    best_score = -1.0

    for height_frac in (0.05, 0.03, 0.02, 0.01, 0.005):
        height_threshold = height_frac * np.max(data_smooth)

        try:
            peaks, _ = find_peaks(
                data_smooth,
                height=height_threshold,
                distance=5,
                prominence=0.005 * np.max(data_smooth),
            )
        except Exception:
            continue

        if len(peaks) < n_liz:
            continue

        selected = np.sort(peaks[:n_liz])
        score = _validate_spacing(selected, dLIZ)

        if score > best_score:
            best_score = score
            best_result = selected

    if best_result is None:
        return None

    # snap к локальному максимуму
    return _snap_peaks_to_local_max(data, best_result, radius=4)


def _snap_peaks_to_local_max(
    data: np.ndarray,
    peaks: np.ndarray,
    radius: int = 4,
) -> np.ndarray:
    """Сдвиг пиков к локальному максимуму."""
    n = len(data)
    corrected = np.copy(peaks)

    for i, pk in enumerate(peaks):
        lo = max(0, int(pk) - radius)
        hi = min(n, int(pk) + radius + 1)
        corrected[i] = lo + int(np.argmax(data[lo:hi]))

    return corrected


def _validate_spacing(
    positions: np.ndarray,
    dLIZ: np.ndarray,
) -> float:
    """Оценка соответствия расстояний."""
    if len(positions) < 2 or len(dLIZ) == 0:
        return 0.0

    actual = np.diff(positions.astype(float))
    n_common = min(len(actual), len(dLIZ))
    if n_common == 0:
        return 0.0

    ratios = actual[:n_common] / np.where(dLIZ[:n_common] == 0, 1e-10, dLIZ[:n_common])

    if len(ratios) < 2:
        return 0.5

    mean_ratio = np.mean(ratios)
    if mean_ratio <= 0:
        return 0.0

    cv = np.std(ratios) / mean_ratio
    return max(0.0, 1.0 - cv)


def _empty_result(raw_ref: np.ndarray):
    """Пустой результат (MATLAB-совместимо)."""
    ref_out = raw_ref[::-1] if len(raw_ref) else raw_ref
    return (
        np.array([], dtype=int),
        np.array([], dtype=np.float64),
        ref_out,
        np.array([], dtype=np.float64),
    )
