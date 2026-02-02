from typing import Any
import numpy as np
from msbackadj import msbackadj
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import trapezoid, quad


def GLFind(data: list[float], peak, sizes: list[float], concentrations: list[float]) -> Any:
    raw_data = np.array(data)
    # 1. Выбор первых 50 значений как шума
    noise = raw_data[0:50]
    # Вычитание шума из данных
    denoised_data = raw_data - np.mean(noise)
    x = np.arange(len(data))
    denoised_data = msbackadj(x, denoised_data, window_size=140, step_size=40, quantile_value=0.1)  # коррекция бейзлайна

    # 2. Обработка данных
    filtered_data = savgol_filter(denoised_data, 5, 1)
    diff_data = np.diff(filtered_data)
    filtered_diff_data = savgol_filter(diff_data, 5, 1)
    diff_diff_data = np.diff(filtered_diff_data)
    filtered_diff_diff_data = savgol_filter(diff_diff_data, 5, 1)

    flipped_data = -filtered_diff_diff_data
    flipped_data[flipped_data < 0] = 0

    # Нахождение пиков
    peak_locations, _ = find_peaks(flipped_data)
    selected_peaks = denoised_data[peak_locations]
    threshold = np.mean(selected_peaks) / 3  # применили порог: береём среднюю высоту по всем найденным пикам и и делим на 3 (иначе слишком высоко)
    selected_peaks = selected_peaks[selected_peaks >= threshold]  # удалили все пики, которые лежат ниже порога

    selected_peak_locations = np.where(np.isin(denoised_data, selected_peaks))[0]
    selected_peak_locations = selected_peak_locations[
        (selected_peak_locations >= 10) & (selected_peak_locations <= (len(x) - 10))
    ]

    if len(selected_peak_locations) == 0:
        # Вывести диалоговое окно с предупреждением
        print('Пики не были найдены', 'Предупреждение', 'warn', 'modal')
        return

    one_pks = np.array([])
    for i, peak_idx in enumerate(selected_peak_locations):
        left_idx = peak_idx - 4
        right_idx = peak_idx + 4

        # Ищем максимум между текующими границами
        max_value = np.max(denoised_data[left_idx:right_idx])  # это по ОУ
        peak_idx = np.where(np.isin(denoised_data, max_value))[0][0]  # это перевод в ОХ

        # Находим границы от текущего максимального значения +-4
        left_idx = peak_idx - 4
        right_idx = peak_idx + 4

        # Значения слева и справа
        left_value = denoised_data[left_idx]
        right_value = denoised_data[right_idx]

        # Проверяем, если обе точки ниже 90% от основного пика (если обе лежат ниже, значит это отдельный пик, а не часть локальных минимумов)
        if left_value < 0.9 * max_value and right_value < 0.9 * max_value:
            # Добавляем в массив standard_pks максимальное значение
            one_pks = np.append(one_pks, max_value)
            selected_peak_locations[i] = peak_idx  # заменяем на найденный максимум
    lonely_pks = np.where(np.isin(denoised_data, one_pks))[0]

    # 3. Нахождение минимумов электрофореграммы
    flip = -denoised_data
    min_peak_locations, _ = find_peaks(flip, distance=8)
    peaks1 = flip[min_peak_locations]

    peaks_threshold = 0.6 * np.mean(flip)

    # Удаление пиков, которые лежат ниже порога
    below_threshold = peaks1 < peaks_threshold
    all_local_minimums = min_peak_locations[below_threshold]  # все минимумы, которые лежат ниже порога (в том числе одиночных пиков)
    min_peak_locations = min_peak_locations[~below_threshold]  # удаляем из массива min_peakLocations все миниуммы, которые лежат ниже порога (оставляем только основные)

    # Объединение найденных пиков и точек пересечения
    crossings2 = np.where(np.diff(denoised_data > 0))

    complete_peaks_locations = np.union1d(min_peak_locations, crossings2)

    # 4. Калибровка данных
    # Инициализируйте переменные
    lib_peak_locations = np.array([])
    hidden_lib_peak_locations = np.array([])

    unrecognized_peaks = np.array([])
    pre_unrecognized_peaks = np.array([])

    hidden_lib_areas = np.array([])
    rest_peaks_areas = np.array([])

    final_lib_local_minimums = np.array([])

    st_areas = np.array([])

    st_peaks = np.array([peak[0], peak[-1]])
    conc = np.array([concentrations[0], concentrations[-1]])

    sdc = np.polyfit(peak, sizes, 5)
    sdc2 = np.polyfit(sizes, peak, 5)

    for i in range(len(lonely_pks) - 1):
        # Вычисление pace
        pace = st_peaks[1] - st_peaks[0]
        # Инициализация массивов
        st_length = [lonely_pks[0]]  # Инициализируем массив st_length первым значением lonely_pks
        # Базовое значение
        base_value = lonely_pks[0]

        # Поиск пиков
        for i in range(1, len(lonely_pks)):
            # Вычитание базового значения
            distance = lonely_pks[i] - base_value
            # Проверка расстоянияs
            if np.abs(distance - pace) <= 0.2 * pace:  # Используем 20% допуск
                st_length.append(lonely_pks[i])
            else:
                pre_unrecognized_peaks = np.append(pre_unrecognized_peaks, lonely_pks[i])

        if len(lonely_pks) > 1 and len(st_length) < 2:
            st_length = [lonely_pks[0], lonely_pks[-1]]

        if len(st_length) > 2:
            # Создаем массив для хранения площадей
            areas = []

            # Перебираем все значения, начиная со второго
            for i in range(1, len(st_length)):
                # Определяем границы (±7)
                left_idx = st_length[i] - 7
                right_idx = st_length[i] + 7

                # Интегрируем площадь между границами
                area = trapezoid(denoised_data[left_idx:right_idx])

                areas.append(area)

            # Находим индекс наибольшей площади
            max_idx = np.argmax(areas)

            # Выбираем соответствующее значение из st_length
            best_value = st_length[max_idx + 1]  # +1 из-за смещения

            # Обновляем st_length
            st_length = [st_length[0], best_value]

    # % В случае, если не было найдено ни одного реперного пика
    # if length(st_length) ~= 2
    #         % Открываем диалоговое окно
    #         choice = questdlg('Реперные пики не найдены.', ...
    #                             'Ошибка анализа', ...
    #                             'Завершить анализ', 'Ввести вручную', 'Ввести вручную');
    #         % Обрабатываем выбор пользователя
    #         switch choice
    #             case 'Завершить анализ'
    #                 % Завершить выполнение анализа
    #                 disp('Анализ завершен пользователем.');
    #                 return;
    #             case 'Ввести вручную'
    #                 [st_length] = plotFigure(denoised_data, st_length);
    #                 selectedPeakLocations = vertcat(selectedPeakLocations, st_length);
    #                 selectedPeakLocations = unique(sort(selectedPeakLocations));
    #         end
    # end

    pre_unrecognized_peaks = np.unique(pre_unrecognized_peaks)
    pre_unrecognized_peaks = pre_unrecognized_peaks[~np.isin(pre_unrecognized_peaks, st_length)]  # если найденный неопознанный пик отнесён к реперному, то он удаляется из массива

    min_st_length = []

    for i in range(len(st_length)):
        # Ищем ближайшее значение в массиве complete_Peaks_Locations
        current_value = st_length[i]

        left_candidates = np.max(complete_peaks_locations[complete_peaks_locations < current_value])
        right_candidates = np.min(complete_peaks_locations[complete_peaks_locations > current_value])
        closest_value = [left_candidates, right_candidates]

        idx = np.abs(closest_value - current_value)

        # Проверяем расстояние с двух сторон
        if np.any(idx > 10):
            # Если хотя бы одно расстояние превышает 10, добавляем значения в min_st_length
            min_st_length.append(current_value - 7)
            min_st_length.append(current_value + 7)

    # Объединяем массивы
    complete_peaks_locations = np.union1d(complete_peaks_locations, min_st_length)
    # На случай, если репер и неопознанный пик находятся слишком близко (расстояние < 10) - иначе впоследствии программа не понимает,
    # какой минимум между ними должен быть (их минимумы накладываются)
    for i in range(len(st_length)):
        current_value = st_length[i]
        # Поиск значений в pre_unrecognized_peaks, лежащих в пределах [current_st_length - 10, current_st_length + 10]
        nearby_peaks = pre_unrecognized_peaks[(pre_unrecognized_peaks >= (current_value - 10)) & (pre_unrecognized_peaks <= (current_value + 10))]

        for k in range(len(nearby_peaks)):
            # Берем первое найденное значение
            target_peak = nearby_peaks[k]

            # Определение диапазона в denoised_data для поиска минимума
            range_start = min(current_value, target_peak)
            range_end = max(current_value, target_peak)

            # Поиск наименьшего значения в указанном диапазоне и его индекса
            min_index = np.argmin(denoised_data[range_start:range_end + 1])
            min_index += range_start  # Коррекция индекса

            # Удаление значений из complete_Peaks_Locations, лежащих в этом диапазоне
            complete_peaks_locations = complete_peaks_locations[
                (complete_peaks_locations < range_start) | (complete_peaks_locations > range_end)
            ]
            # Добавление найденного индекса в complete_Peaks_Locations
            complete_peaks_locations = np.union1d(complete_peaks_locations, min_index)

    # Удаляем пики, которые лежат за пределами реперов
    selected_peak_locations = selected_peak_locations[(selected_peak_locations >= st_length[0]) & (selected_peak_locations <= st_length[-1])]

    # Этот блок приводит электрофореграмму геномной библиотеки и стандарта
    # длин в одну шкалу (выравнивает по ширине и высоте)
    px = np.polyfit(st_length, st_peaks, 1)  # выравнивание по ширине
    t = np.arange(len(denoised_data))
    t_main = np.polyval(px, t)

    x_fill_1 = np.array([])
    x_lib_fill_1 = np.array([])
    y_fill = np.array([])

    # 5. Обработка данных с учётом калибровки
    # В этом блоке теперь находим и разбиваем все локальные пики по классам: реперные пики, пики геномной библиотеки и неопознанные пики
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
                    x_vals = np.arange(len(denoised_data))
                    # Получаем соответствующие y-координаты, используя интерполяцию
                    y_fill = np.interp(x_fill_1, x_vals, denoised_data)
                elif np.any(np.isin(x_range, st_length)):
                    x_vals = np.arange(len(denoised_data))
                    area, _ = quad(lambda x: np.interp(x, x_vals, denoised_data), x_range[0], x_range[-1])  # реперный пик
                    st_areas = np.append(st_areas, area)
        elif num_points_between_peaks >= 4:
            # Найдем максимальное значение denoised_data между текущими точками
            # Получаем индексы для complete_Peaks_Locations
            start_index = complete_peaks_locations[i]
            end_index = complete_peaks_locations[i + 1]
            f = complete_peaks_locations[i + 1]

            # Находим максимальное значение между этими индексами
            max_value_lib = np.max(denoised_data[start_index:end_index + 1])
            max_lib_value = np.where(denoised_data == max_value_lib)[0]
            max_lib_value_corr = np.polyval(sdc, max_lib_value)

            lower_bound = np.polyval(sdc2, max_lib_value_corr - 200)
            upper_bound = np.polyval(sdc2, max_lib_value_corr + 200)

            # Проверяем, лежит ли диапазон внутри [start_index, end_index]
            if lower_bound > start_index or upper_bound < end_index:
                # Если диапазон выходит за пределы, создаем плавный диапазон
                x_lib_fill_1 = np.linspace(start_index, end_index, 100)  # Разбиваем на 100 точек для плавности

                # Интерполируем значения y для x_fill_1
                x_vals = np.arange(len(denoised_data))
                y_lib_fill = np.interp(x_lib_fill_1, x_vals, denoised_data)

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            final_lib_local_minimums = np.concatenate((final_lib_local_minimums, selected_peak_locations[indices_between_peaks], [f]))  # берём текущую пару точек вместо локальных минимумов (теперь это просто один большой пик ГБ)

            # Ищем локальные максимумы между найденными локальными минимумами final_Lib_local_minimums
            for j in range(len(final_lib_local_minimums) - 1):
                # Ищем индексы массива denoised_data, которые находятся между final_Lib_local_minimums(i) и final_Lib_local_minimums(i+1)
                x_start = final_lib_local_minimums[j]
                x_end = final_lib_local_minimums[j + 1]
                # Находим соответствующие индексы в массиве denoised_data
                indices = np.where((np.arange(len(denoised_data)) >= x_start) & (np.arange(len(denoised_data)) <= x_end))[0]

                if len(indices) > 0:
                    # Находим максимальное значение в массиве denoised_data между этими индексами
                    max_value = np.max(denoised_data[indices])
                    indices_max_value = np.where(denoised_data == max_value)[0]
                    # Добавляем это максимальное значение в LibPeakLocations
                    lib_peak_locations = np.append(lib_peak_locations, indices_max_value)

            lib_peak_locations = np.unique(lib_peak_locations)
            # На случай, если алгоритм посчитал тонкие ГБ (типа фаикс) в качестве большой библиотеки
            diff_values = np.abs(denoised_data[lib_peak_locations.astype(int)] - max_value_lib)  # вычитаем из найденных пиков ГБ максимальный пик
            # Определяем порог 20% от maxLibValue - расстояние (высота) между пиками не должно быть больше, чем 20% от
            # максимального пика: максимум на 100, минимум на 50, расстояние между ними 50 > 20 - это плохо (+1 на следующем шагу);
            # минимум на 90, расстояние 10 < 20 - хорошо (0)
            threshold = 0.2 * max_value_lib
            # Проверяем, какие значения превышают порог и заполняем check_LibPeaks
            check_lib_peaks = diff_values > threshold  # ищем пики, которые расположены слишком низко (ниже, чем 20% от максимального пика): если больше порога, то возвращается 1 (находим их количество)
            # Считаем разницу между длиной LibPeakLocations и количеством "плохих" значений
            diff_count = len(lib_peak_locations) - np.sum(check_lib_peaks)

            # Если больше 50% значений превышают порог, удаляем массивы
            if diff_count < 0.3 * len(lib_peak_locations):
                lib_peak_locations = np.array([])
            else:
                hidden_lib_peak_locations, hidden_lib_areas, hidden_final_lib_local_minimums = hid_fun(
                    denoised_data, start_index, end_index, px, hidden_lib_areas, max_lib_value
                )
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Подсчёт концентраций и молярности по реперам (ГБ будет дальше)
    main_corr = np.polyval(sdc, t_main)
    st_peaks_corr = np.polyval(sdc, st_peaks)

    st_areas = np.array([st_areas[0], st_areas[-1]])
    led_one_area = st_areas / (st_peaks_corr / 100)  # считает корректно, проверено (в Matlab)
    a = np.polyfit(led_one_area, conc, 1)

    st_molarity = ((conc * 1e-3) / (649 * st_peaks_corr)) * 1e9

    # Если ГБ гладкая/фаикс/слишком низкая
    if len(hidden_lib_peak_locations) == 0:
        rest_peaks = np.copy(selected_peak_locations)
        rest_peaks = rest_peaks[~((rest_peaks >= (st_length[0] - 10)) & (rest_peaks <= (st_length[0] + 10)) | (rest_peaks >= (st_length[1] - 10)) & (rest_peaks <= (st_length[1] + 10)))]

        rest_peaks_locations = np.concatenate((complete_peaks_locations, all_local_minimums))
        rest_peaks_locations = np.sort(rest_peaks_locations)

        i = 0
        while i < len(rest_peaks_locations) - 1:
            indices_between_peaks = (rest_peaks >= rest_peaks_locations[i]) & (rest_peaks <= rest_peaks_locations[i + 1])
            peaks_between = rest_peaks[indices_between_peaks]

            if len(peaks_between) > 1:
                new_peak = np.mean(peaks_between[:2])
                rest_peaks_locations = np.unique(np.append(rest_peaks_locations, new_peak))
                i -= 1
            elif len(peaks_between) == 1:
                x_points = rest_peaks_locations[i:i + 2]

                for j in range(len(x_points) - 1):
                    x_range = np.arange(x_points[j], x_points[j + 1] + 1)
                    x_vals = np.arange(len(denoised_data))
                    area, _ = quad(lambda x: np.interp(x, x_vals, denoised_data), x_range[0], x_range[-1])
                    rest_peaks_areas = np.append(rest_peaks_areas, area)

            i += 1

        max_index = np.argmax(rest_peaks_areas)
        lib_peak_locations = rest_peaks[max_index]

        start_index = max(rest_peaks_locations[rest_peaks_locations < lib_peak_locations])
        end_index = min(rest_peaks_locations[rest_peaks_locations > lib_peak_locations])

        hidden_lib_peak_locations, hidden_lib_areas, hidden_final_lib_local_minimums = hid_fun(
            denoised_data, start_index, end_index, px, hidden_lib_areas, lib_peak_locations)

        unrecognized_peaks = np.copy(rest_peaks)
        unrec_areas = np.copy(rest_peaks_areas)

        unrecognized_peaks = np.delete(unrecognized_peaks, max_index)
        unrec_areas = np.delete(unrec_areas, max_index)

    hid_lib_length = np.polyval(px, hidden_lib_peak_locations)
    hid_lib_peaks_corr = np.polyval(sdc, hid_lib_length)
    hid_one_area = hidden_lib_areas / (hid_lib_peaks_corr / 100)
    hid_one_area_conc = np.polyval(a, hid_one_area)
    hid_molarity = ((hid_one_area_conc * 1e-3) / (649 * hid_lib_peaks_corr)) * 1e9

    lib_areas = np.array([])
    lib_one_area = np.array([])
    lib_one_area_conc = np.array([])
    lib_molarity = np.array([])

    if len(rest_peaks_areas) == 0:
        check_lib_peak_locations = lib_peak_locations[(lib_peak_locations < hidden_lib_peak_locations[0]) | (lib_peak_locations > hidden_lib_peak_locations[-1])]
        if len(check_lib_peak_locations) > 0:
            lib_peak_locations = np.setdiff1d(lib_peak_locations, check_lib_peak_locations)
            if np.any(check_lib_peak_locations < hidden_lib_peak_locations[0]):
                lib_peak_locations = np.insert(lib_peak_locations, 0, hidden_lib_peak_locations[0])
                left_candidates = max(final_lib_local_minimums[final_lib_local_minimums < lib_peak_locations[0]])
                final_lib_local_minimums = final_lib_local_minimums[final_lib_local_minimums >= left_candidates]
            if np.any(check_lib_peak_locations > hidden_lib_peak_locations[-1]):
                lib_peak_locations = np.append(lib_peak_locations, hidden_lib_peak_locations[-1])
                right_candidates = min(final_lib_local_minimums[final_lib_local_minimums > lib_peak_locations[-1]])
                final_lib_local_minimums = final_lib_local_minimums[final_lib_local_minimums <= right_candidates]

        lib_peak_locations = np.unique(np.sort(lib_peak_locations))
        lib_length = np.polyval(px, lib_peak_locations)
        lib_peaks_corr = np.polyval(sdc, lib_length)

        indices = np.searchsorted(hidden_lib_peak_locations, lib_peak_locations)
        prev_idx = 0
        for i in indices:
            lib_areas = np.append(lib_areas, np.sum(hidden_lib_areas[prev_idx:i]))
            lib_one_area = np.append(lib_one_area, np.sum(hid_one_area[prev_idx:i]))
            lib_one_area_conc = np.append(lib_one_area_conc, np.sum(hid_one_area_conc[prev_idx:i]))
            lib_molarity = np.append(lib_molarity, np.sum(hid_molarity[prev_idx:i]))
            prev_idx = i + 1

    elif len(rest_peaks_areas) > 0:
        lib_length = np.polyval(px, lib_peak_locations)
        lib_peaks_corr = np.polyval(sdc, lib_length)
        max_lib_value = lib_peak_locations
        lib_areas = np.append(lib_areas, np.sum(hidden_lib_areas))
        lib_one_area = np.append(lib_one_area, np.sum(hid_one_area))
        lib_one_area_conc = np.append(lib_one_area_conc, np.sum(hid_one_area_conc))
        lib_molarity = np.append(lib_molarity, np.sum(hid_molarity))

    # one_area = np.concatenate(([led_one_area[0]], lib_one_area, [led_one_area[-1]]))  # площадь на один фрагмент, не нужен в коде, но может понадобиться для проверки!!!
    all_areas_conc = np.concatenate(([concentrations[0]], lib_one_area_conc, [concentrations[-1]]))  # концентрация
    all_areas = np.concatenate(([st_areas[0]], lib_areas, [st_areas[-1]]))  # общая площадь фрагмента
    molarity = np.concatenate(([st_molarity[0]], lib_molarity, [st_molarity[-1]]))  # молярность

    all_peaks = np.concatenate(([st_length[0]], lib_peak_locations, [st_length[-1]]))  # время выхода
    all_peaks_corr = np.concatenate(([st_peaks_corr[0]], lib_peaks_corr, [st_peaks_corr[-1]]))  # длина в пн

    t_final_locations = np.polyval(px, final_lib_local_minimums)
    t_unrecognized_peaks = np.polyval(px, unrecognized_peaks)  # пересчёт по времени неизвестных пиков
    unrecognized_peaks_corr = np.polyval(sdc, t_unrecognized_peaks)  # только неопознанные пики
    max_lib_peak = main_corr[max_lib_value]  # максимальный пик библиотеки

    total_lib_area = np.sum(lib_areas)
    total_lib_conc = np.sum(lib_one_area_conc)
    total_lib_molarity = np.sum(lib_molarity)

    # Закрашиваем красным ложные пики и широкую библиотеку
    if len(x_fill_1):
        x_fill_1 = np.array([x_fill_1[0], x_fill_1[-1]])
        x_fill_1 = t_main[x_fill_1.astype(int)]
        x_fill = np.linspace(x_fill_1[0], x_fill_1[-1], 100)
    else:
        x_fill = np.array([])

    if len(x_lib_fill_1):
        x_lib_fill_1 = np.array([x_lib_fill_1[0], x_lib_fill_1[-1]])
        x_lib_fill_1 = t_main[x_lib_fill_1.astype(int)]
        x_lib_fill = np.linspace(x_lib_fill_1[0], x_lib_fill_1[-1], 100)
    else:
        x_lib_fill = np.array([])

    hpx = np.round(lib_peaks_corr)
    unr = np.round(unrecognized_peaks_corr)
    stp = np.array([sizes[0], sizes[-1]])
    return (
        t_main, denoised_data, st_peaks, st_length, t_unrecognized_peaks, unrecognized_peaks,
        lib_length, lib_peak_locations, t_final_locations, final_lib_local_minimums,
        hpx, unr, stp, main_corr, all_areas, all_peaks_corr, all_peaks, all_areas_conc, molarity,
        max_lib_peak, max_lib_value, total_lib_area, total_lib_conc, total_lib_molarity, x_fill, y_fill, x_lib_fill, y_lib_fill
    )


def hid_fun(denoised_data, start_index, end_index, px, hidden_lib_areas, max_lib_value) -> Any:

    median_before_max = np.median(np.arange(start_index, max_lib_value + 1))  # Находим медианное значение между start_index и maxLibValue (левая середина пика ГБ)
    median_after_max = np.median(np.arange(max_lib_value, end_index + 1))  # Находим медианное значение между maxLibValue и end_index (правая середина пика ГБ)

    # Создаем массив из двух медианных значений и округляем их
    rounded_values = np.round([median_before_max, median_after_max]).astype(int)

    # Создаем массив чисел от первого округленного значения до второго с шагом 1
    hidden_lib_peak_locations = np.arange(rounded_values[0], rounded_values[1] + 1)
    hidden_final_lib_local_minimums = np.concatenate(([start_index], hidden_lib_peak_locations))  # все площади

    x_vals = np.arange(len(denoised_data))

    def interp_func(x):
        return np.interp(x, x_vals, denoised_data)

    # ДЛЯ ЗАКРАСКИ И ОБЩЕЙ ПЛОЩАДИ
    for i in range(len(hidden_final_lib_local_minimums) - 1):
        h_x_range1 = np.arange(hidden_final_lib_local_minimums[i], hidden_final_lib_local_minimums[i + 1] + 1)
        area, _ = quad(interp_func, h_x_range1[0], h_x_range1[-1])
        hidden_lib_areas = np.append(hidden_lib_areas, area)
    return hidden_lib_peak_locations, hidden_lib_areas, hidden_final_lib_local_minimums
