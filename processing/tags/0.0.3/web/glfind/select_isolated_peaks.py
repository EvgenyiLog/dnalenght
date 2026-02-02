import numpy as np
from numpy.typing import NDArray


def select_isolated_peaks(
    peaks: NDArray[np.integer],
    corrected_signal: NDArray[np.floating],
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    """Выделение одиночных пиков и уточнение местоположения имеющихся"""

    isolated_peaks = []
    # Мы не мутируем аргумент, а возвращаем новый массив
    refined_peaks = np.copy(peaks)
    # Проходим по каждому пику в selectedPeakLocations
    for i, peak_idx in enumerate(peaks):
        search_start = peak_idx - 4
        search_end = peak_idx + 4

        # Ищем индекс максимума между текущими границами и его границы
        local_window = corrected_signal[search_start:search_end + 1]
        local_max_idx = search_start + np.argmax(local_window)

        # Значения слева, справа и в пике
        flank_left = corrected_signal[local_max_idx - 4]
        flank_right = corrected_signal[local_max_idx + 4]
        peak_value = corrected_signal[local_max_idx]

        # Проверяем, если обе точки ниже 90% от основного пика (если обе лежат ниже, значит это отдельный пик, а не часть локальных минимумов)
        if flank_left < 0.93 * peak_value and flank_right < 0.93 * peak_value:
            refined_peaks[i] = local_max_idx  # заменяем на найденный максимум
            isolated_peaks.append(local_max_idx)  # Добавляем в массив standard_pks максимальное значение

    return np.array(isolated_peaks, dtype=np.int64), refined_peaks
