from lib.glfind.compute_hidden_library_area import compute_hidden_library_area


import numpy as np
from numpy.typing import NDArray

from lib.glfind.select_isolated_peaks import select_isolated_peaks


def classify_and_extract_library_peaks(
    corrected_signal: NDArray[np.floating],
    selected_peak_locations: NDArray[np.integer],
    reference_peaks: NDArray[np.integer],
    complete_peaks_locations: NDArray[np.integer],
    pre_unrecognized_peaks: NDArray[np.integer],
    sdc: NDArray[np.floating],
    sdc2: NDArray[np.floating],
) -> tuple[
    NDArray[np.integer],
    NDArray[np.integer],
    NDArray[np.floating],
    NDArray[np.integer],
    NDArray[np.integer],
    np.integer,
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """Обработка данных с учётом калибровки: классификация пиков и извлечение информации о библиотеке"""

    lib_peak_locations = np.empty(0, dtype=np.int64)
    hidden_lib_peak_locations = np.empty(0, dtype=np.int64)
    unrecognized_peaks = np.empty(0, dtype=np.int64)
    final_lib_local_minimums = np.empty(0, dtype=np.int64)
    hidden_lib_areas = np.empty(0, dtype=np.float64)
    x_fill_1 = np.empty(0, dtype=np.float64)
    x_lib_fill_1 = np.empty(0, dtype=np.float64)
    y_fill = np.empty(0, dtype=np.float64)
    y_lib_fill = np.empty(0, dtype=np.float64)
    st_areas: list[float] = []

    max_lib_value = np.int64(-1)

    for i in range(len(complete_peaks_locations) - 1):
        # Как обычно проверяем наличие локальных максимумов между текущей парой локальных минимумов
        indices_between_peaks = (selected_peak_locations >= complete_peaks_locations[i]) & (selected_peak_locations <= complete_peaks_locations[i + 1])
        # Считаем количество точек, попавших в пару локальных минимумов
        num_points_between_peaks = np.sum(indices_between_peaks)
        if 0 < num_points_between_peaks < 4:
            # Отмечаем пространство между текущими точками
            x_points = complete_peaks_locations[i:i + 2]
            # Для подсчёта площадей
            for j in range(len(x_points) - 1):
                # Выделяем текущую область
                x_range = np.arange(x_points[j], x_points[j + 1] + 1)
                # y_range = denoised_data[x_range]
                # Проверка наличия значений из unrecognized_peaks в x_range
                if np.any(np.isin(x_range, unrecognized_peaks)):  # неопознанный пик
                    unrecognized_peaks = np.append(unrecognized_peaks, pre_unrecognized_peaks[j])
                    # Определяем x-координаты области
                    x_fill_1 = np.linspace(x_range[0], x_range[-1], 100)  # Разбиваем на 100 точек для плавности
                    x_vals = np.arange(len(corrected_signal))
                    # Получаем соответствующие y-координаты, используя интерполяцию
                    y_fill = np.interp(x_fill_1, x_vals, corrected_signal)
                elif np.any(np.isin(x_range, reference_peaks)):
                    start_idx = x_range[0]
                    end_idx = x_range[-1]
                    area = float(np.trapezoid(corrected_signal[start_idx:end_idx + 1]))
                    st_areas.append(area)
        elif num_points_between_peaks >= 4:
            # Найдем максимальное значение denoised_data между текущими точками
            # Получаем индексы для complete_Peaks_Locations
            start_index = complete_peaks_locations[i]
            end_index = complete_peaks_locations[i + 1]
            f = complete_peaks_locations[i + 1]

            # Находим максимальное значение между этими индексами
            max_lib_value = start_index + np.argmax(corrected_signal[start_index:end_index + 1])
            max_value_lib = corrected_signal[max_lib_value]
            max_lib_value_corr = np.polyval(sdc, max_lib_value)

            lower_bound = np.polyval(sdc2, max_lib_value_corr - 200)
            upper_bound = np.polyval(sdc2, max_lib_value_corr + 200)

            # Проверяем, лежит ли диапазон внутри [start_index, end_index]
            if lower_bound > start_index or upper_bound < end_index:
                # Если диапазон выходит за пределы, создаем плавный диапазон
                x_lib_fill_1 = np.linspace(start_index, end_index, 100)  # Разбиваем на 100 точек для плавности

                # Интерполируем значения y для x_fill_1
                x_vals = np.arange(len(corrected_signal))
                y_lib_fill = np.interp(x_lib_fill_1, x_vals, corrected_signal)

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            final_lib_local_minimums = np.concatenate((final_lib_local_minimums, selected_peak_locations[indices_between_peaks], [f]))  # берём текущую пару точек вместо локальных минимумов (теперь это просто один большой пик ГБ)

            # Ищем локальные максимумы между найденными локальными минимумами final_Lib_local_minimums
            for j in range(len(final_lib_local_minimums) - 1):
                # Ищем индексы массива denoised_data, которые находятся между final_Lib_local_minimums(i) и final_Lib_local_minimums(i+1)
                x_start = final_lib_local_minimums[j]
                x_end = final_lib_local_minimums[j + 1]
                # Находим соответствующие индексы в массиве denoised_data
                indices = np.where((np.arange(len(corrected_signal)) >= x_start) & (np.arange(len(corrected_signal)) <= x_end))[0]

                if len(indices) > 0:
                    # Находим максимальное значение в массиве denoised_data между этими индексами
                    max_value = np.max(corrected_signal[indices])
                    indices_max_value = np.where(corrected_signal == max_value)[0]
                    # Добавляем это максимальное значение в LibPeakLocations
                    lib_peak_locations = np.append(lib_peak_locations, indices_max_value)

            lib_peak_locations = np.unique(lib_peak_locations)

            # На случай, если алгоритм посчитал тонкие ГБ (типа фаикс) в качестве большой библиотеки
            diff_values = np.abs(corrected_signal[lib_peak_locations.astype(int)] - max_value_lib)  # вычитаем из найденных пиков ГБ максимальный пик

            # Определяем порог 20% от maxLibValue - расстояние (высота) между пиками не должно быть больше, чем 20% от
            # максимального пика: максимум на 100, минимум на 50, расстояние между ними 50 > 20 - это плохо (+1 на следующем шагу);
            # минимум на 90, расстояние 10 < 20 - хорошо (0)
            threshold = 0.2 * max_value_lib

            # Проверяем, какие значения превышают порог и заполняем check_LibPeaks
            check_lib_peaks = diff_values > threshold  # ищем пики, которые расположены слишком низко (ниже, чем 20% от максимального пика): если больше порога, то возвращается 1 (находим их количество)

            # Считаем разницу между длиной LibPeakLocations и количеством "плохих" значений
            diff_count = len(lib_peak_locations) - np.sum(check_lib_peaks)

            checking_pks = select_isolated_peaks(lib_peak_locations, corrected_signal)[0]

            # Если больше 50% значений превышают порог, удаляем массивы
            if diff_count < 0.3 * len(lib_peak_locations) and len(checking_pks) == 1:
                lib_peak_locations = np.empty(0, dtype=np.int64)
            else:
                hidden_lib_peak_locations, new_hidden_lib_areas, _ = compute_hidden_library_area(
                    corrected_signal, start_index, end_index, max_lib_value
                )
                hidden_lib_areas = np.append(hidden_lib_areas, new_hidden_lib_areas)
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    return (
        lib_peak_locations,
        hidden_lib_peak_locations,
        hidden_lib_areas,
        final_lib_local_minimums,
        unrecognized_peaks,
        max_lib_value,
        np.array(st_areas, dtype=np.float64),
        x_fill_1,
        x_lib_fill_1,
        y_fill,
        y_lib_fill,
    )
