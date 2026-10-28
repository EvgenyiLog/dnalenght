import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


def find_signal_minima(
    corrected_signal: NDArray[np.floating],
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    """Нахождение минимумов электрофореграммы"""

    inverted_signal = -corrected_signal
    strong_minima_indices = find_peaks(inverted_signal, distance=9)[0]  # Equal MinPeakDistance=8
    minima_heights = inverted_signal[strong_minima_indices]

    threshold = 0.6 * np.mean(inverted_signal)

    # Удаление пиков, которые лежат ниже порога
    weak_minima_mask = minima_heights < threshold
    weak_minima_indices = strong_minima_indices[weak_minima_mask]  # все минимумы, которые лежат ниже порога (в том числе одиночных пиков)
    strong_minima_indices = strong_minima_indices[~weak_minima_mask]  # удаляем из массива min_peakLocations все миниуммы, которые лежат ниже порога (оставляем только основные)

    # Объединение найденных пиков и точек пересечения
    zero_crossings = np.where(np.diff(corrected_signal > 0))[0]
    combined_minima = np.union1d(strong_minima_indices, zero_crossings)

    return combined_minima, weak_minima_indices
