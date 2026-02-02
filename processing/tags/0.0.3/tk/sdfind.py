from typing import Any
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import quad
from wden2 import wden
from msbackadj import msbackadj


# rawRef - data
# InLIZ - sizes
# locs_pol - release_times
# CONC - concentrations
def SDFind(data: list[float], in_sizes: list[float], release_times: list[int], concentrations: list[float]) -> Any:
    sizes = np.flip(in_sizes)
    x = np.arange(len(data))
    raw_data = msbackadj(x, data, window_size=140, step_size=40, quantile_value=0.1)  # коррекция бейзлайна
    filtered_data = wden(raw_data, 'sqtwolog', 's', 'sln', 1, 'sym2')  # фильтр данных

    filtered_data = np.flip(filtered_data)
    raw_data = np.flip(raw_data)

    # Понадобится для отсеивания найденных пиков в соотвествии с выбранным законом
    poly_coef = np.polyfit(in_sizes, release_times, 4)  # полином 4 степени
    new_sizes = np.polyval(poly_coef, sizes)
    d_sizes = np.abs(np.diff(new_sizes))

    threshold = np.quantile(filtered_data, 0.995)  # для начала возьмем порог на уровне 99.5%, будем его снижать, если надо

    # *** НАЙДЕМ В СПЕКТРЕ ПИКИ, СООТВЕТВУЮЩИЕ ПИКАМ СТАНДАРТА ***
    for _ in range(30):   # главный цикл (30 попыток)
        # ищем пики, пытаемся среди найденных отобрать подходящие, если не
        # удалось - снижаем порог и повторяем процедуру

        # ** ИЩЕМ ПИКИ В СПЕКТРЕ **
        threshold *= 0.9

        for _ in range(20):
            peaks, _ = find_peaks(filtered_data, height=threshold, distance=8)
            if len(peaks) < len(sizes):
                threshold *= 0.9
            else:
                break

        overmuch = 2.4  # порог, значение взято из опыта
        if len(peaks) >= overmuch * len(sizes):
            return [], [], [], data, []

        for k in range(len(sizes) - 1):
            for j in range(k + 1, len(peaks)):
                pace = (peaks[j] - peaks[k]) / d_sizes[0]  # кандидат на "базовый шаг"
                data_peak_idx = [0]  # кандидат

                i_liz, i_dat, d_next = 0, 0, 0
                while d_next < peaks[-1] and i_liz < len(sizes) - 1:
                    d_prev = peaks[i_dat]
                    d_d = pace * d_sizes[i_liz]
                    d_next = d_prev + d_d
                    dst = np.abs(peaks - d_next)
                    idx = np.argmin(dst)   # индекс ближайшего значения
                    minmin = dst[idx]

                    if minmin < d_d / 2:  # пик лежит примерно там где и ожидалось
                        data_peak_idx.append(int(idx))
                        pace = (peaks[idx] - d_prev) / d_sizes[i_liz]  # фактический шаг
                        i_liz += 1
                        i_dat = int(idx)
                    else:  # пика в ожидаемом месте нет - возможно начальный пик ложный
                        i_dat = data_peak_idx[0] + 1  # примем следующий пик за начальный
                        data_peak_idx = [i_dat]
                        pace = (peaks[j] - peaks[k]) / d_sizes[0]
                        i_liz = 0  # стандарт начнем с начала

                if len(data_peak_idx) == len(sizes):  # нужное количество пиков нашли - выходим из цикла
                    break
            if len(data_peak_idx) == len(sizes):  # нужное количество пиков нашли - выходим из цикла
                break
        if len(data_peak_idx) == len(sizes):  # нужное количество пиков нашли - выходим из цикла
            break

    if len(data_peak_idx) == len(sizes):
        selected_peaks = peaks[data_peak_idx]  # отобранные пики
        #  Нахождение минимумов
        filtered_filtered_data = savgol_filter(filtered_data, 3, 1)
        crossings = np.where(np.diff(filtered_filtered_data > 0))[0]
        flipped_data = -filtered_filtered_data
        peak_locs, _ = find_peaks(flipped_data)
        crossings = np.union1d(peak_locs, crossings)
    else:
        return [], [], [], raw_data, []

    areas = []

    # *** нарисуем что нашли ***
    if len(selected_peaks) == len(sizes):
        # строим площади под графиком и считаем их
        for i in range(len(crossings) - 1):
            # ищем индексы между текущей парой точек
            indices_between = (selected_peaks >= crossings[i]) & (selected_peaks <= crossings[i + 1])
            # считаем количество значений между текущей парой точек
            if np.sum(indices_between) == 1:
                # Выделим текущую область
                x_range = np.arange(crossings[i], crossings[i + 1])
                y_range = raw_data[x_range]
                # Убедимся, что размерности совпадают
                if len(x_range) == len(y_range):
                    x_vals = np.arange(len(data))
                    area_val, _ = quad(lambda x: np.interp(x, x_vals, data), x_range[0], x_range[-1])
                    areas.append(area_val)

        np_areas = np.flip(areas)
        # *** развернем обратно вектор обратно, чтобы индексы шли слева направо ***
        raw_data = np.flip(raw_data)
        selected_peaks = -selected_peaks + len(data) - 1
        selected_peaks = np.flip(selected_peaks)

        frg_area = np_areas / (np.flip(sizes) ** 2 / 100)
        np_areas = np_areas * 1e-7

        sd_molarity = ((np.array(concentrations) * 1e-3) / (649 * np.flip(sizes))) * 1e9

        return selected_peaks, np_areas, frg_area, raw_data, sd_molarity

    return [], [], [], raw_data, []
