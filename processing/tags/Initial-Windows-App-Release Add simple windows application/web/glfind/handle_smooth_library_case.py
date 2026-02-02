import numpy as np
from numpy.typing import NDArray

from lib.glfind.compute_hidden_library_area import compute_hidden_library_area


def handle_smooth_library_case(
    corrected_signal: NDArray[np.floating],
    selected_peak_locations: NDArray[np.integer],
    reference_peaks: NDArray[np.integer],
    complete_peaks_locations: NDArray[np.integer],
    all_local_minimums: NDArray[np.integer],
) -> tuple[
    NDArray[np.integer],
    NDArray[np.integer],
    NDArray[np.floating],
    NDArray[np.integer],
    NDArray[np.integer],
    np.integer,
]:
    """Обработка случая, когда геномная библиотека гладкая или содержит один невыраженный пик"""
    rest_peaks_areas: list[float] = []
    rest_peaks = np.copy(selected_peak_locations)  # Собираем все найденные пики
    # Удаляем значения из rest_peaks, которые попадают в диапазон реперных пиков
    rest_peaks = rest_peaks[~((rest_peaks >= (reference_peaks[0] - 10)) & (rest_peaks <= (reference_peaks[0] + 10)) | (rest_peaks >= (reference_peaks[1] - 10)) & (rest_peaks <= (reference_peaks[1] + 10)))]

    # Инициализация массива для хранения площадей
    rest_peaks_locations = np.union1d(complete_peaks_locations, all_local_minimums)

    # Проходим по парам пиков
    i = 0
    while i < len(rest_peaks_locations) - 1:
        # Индексы точек между текущей парой пиков
        indices_between_peaks = (rest_peaks >= rest_peaks_locations[i]) & (rest_peaks <= rest_peaks_locations[i + 1])
        # Точки между пиками
        peaks_between = rest_peaks[indices_between_peaks]

        if len(peaks_between) > 1:
            new_peak = np.int64(np.mean(peaks_between[:2]))  # Среднее значение rest_peaks между двумя соседствующими пиками (минимум между двумя текущими значениями)
            rest_peaks_locations = np.union1d(rest_peaks_locations, new_peak)  # Добавляем новый минимум: если между текущими минимумами нашли несколько максимумов, добавляем новые минимумы, чтобы разделить эти максимумы
            i -= 1
        elif len(peaks_between) == 1:
            # Определяем границы текущей области
            start_idx = rest_peaks_locations[i]
            end_idx = rest_peaks_locations[i + 1]

            area = float(np.trapezoid(corrected_signal[start_idx:end_idx + 1]))
            rest_peaks_areas.append(area)

        i += 1

    # Находим индекс наибольшей площади
    max_index = np.argmax(rest_peaks_areas)
    max_lib_value = rest_peaks[max_index]

    # Записываем соответствующий пик в LibPeakLocations
    lib_peak_locations = np.array([max_lib_value], dtype=np.int64)

    # Находим минимумы, которые принадлежат найденному максимому
    start_index = max(rest_peaks_locations[rest_peaks_locations < lib_peak_locations])
    end_index = min(rest_peaks_locations[rest_peaks_locations > lib_peak_locations])

    hidden_lib_peak_locations, new_hidden_lib_areas, hidden_final_lib_local_minimums = compute_hidden_library_area(
        corrected_signal, start_index, end_index, max_lib_value)

    # Определяем "неопознанные" пики и их площади
    unrecognized_peaks = np.copy(rest_peaks)

    # Удаляем главный пик и его площадь из "неопознанных"
    unrecognized_peaks = np.delete(unrecognized_peaks, max_index)

    final_lib_local_minimums = np.array([hidden_final_lib_local_minimums[0], hidden_final_lib_local_minimums[-1]], dtype=np.int64)

    return (
        lib_peak_locations,
        hidden_lib_peak_locations,
        new_hidden_lib_areas,
        final_lib_local_minimums,
        unrecognized_peaks,
        max_lib_value,
    )
