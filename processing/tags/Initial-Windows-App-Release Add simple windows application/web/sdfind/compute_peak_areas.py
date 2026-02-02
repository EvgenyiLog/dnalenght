import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks, savgol_filter


def compute_peak_areas(
    corrected_signal: NDArray[np.floating],
    denoised_signal: NDArray[np.floating],
    matching_peaks: NDArray[np.integer]
) -> NDArray[np.floating]:
    """Вычисляет площади под сигналом между границами пиков."""

    #  Нахождение минимумов
    smoothed = savgol_filter(denoised_signal, 3, 1)
    zero_crossings = np.where(np.diff(smoothed > 0))[0]
    local_minima = find_peaks(-smoothed)[0]
    split_points = np.union1d(local_minima, zero_crossings)

    # *** нарисуем что нашли ***
    # строим площади под графиком и считаем их
    peak_areas: list[float] = []
    for i in range(len(split_points) - 1):
        # ищем индексы между текущей парой точек
        peaks_between = (matching_peaks >= split_points[i]) & (matching_peaks <= split_points[i + 1])
        # считаем количество значений между текущей парой точек
        if np.sum(peaks_between) == 1:
            # Выделим текущую область
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            area = float(np.trapezoid(corrected_signal[start_idx:end_idx + 1]))
            peak_areas.append(area)

    return np.array(peak_areas, dtype=np.float64)
