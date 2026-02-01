import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter, find_peaks, peak_widths, peak_prominences


def find_significant_peaks2(
    corrected_signal: NDArray[np.floating],
    standard_peaks: NDArray[np.integer]
) -> tuple[NDArray[np.integer], NDArray[np.integer], NDArray[np.integer]]:
    """Обнаружение значимых пиков 2"""

    smoothed = savgol_filter(corrected_signal, 5, 1)
    first_derivative = np.diff(smoothed)
    first_derivative_smoothed = savgol_filter(first_derivative, 5, 1)
    second_derivative = np.diff(first_derivative_smoothed)
    second_derivative_smoothed = savgol_filter(second_derivative, 5, 1)

    peak_candidate_signal = -second_derivative_smoothed
    peak_candidate_signal[peak_candidate_signal < 0] = 0

    # Нахождение пиков
    peak_indices = find_peaks(peak_candidate_signal)[0]
    peak_heights = peak_candidate_signal[peak_indices]
    peak_proms = peak_prominences(peak_candidate_signal, peak_indices)[0]
    peak_width = peak_widths(peak_candidate_signal, peak_indices)[0]

    # Удаляем все крайние пики, найденные на прошлом шаге, из всех массивов
    valid_mask = (peak_indices > 15) & (peak_indices < (len(corrected_signal) - 10))

    peak_indices = peak_indices[valid_mask]
    peak_heights = peak_heights[valid_mask]
    peak_width = peak_width[valid_mask]
    peak_proms = peak_proms[valid_mask]

    # === Points ===
    Points = np.zeros_like(peak_proms, dtype=float)

    # === ЭТАП 1. Работа с разрядами (пики с наибольшим разрядом (единицей) получают +1) ===
    orders = np.floor(np.log10(np.abs(peak_proms))).astype(int)
    max_order = np.max(orders)

    r_peak_proms = peak_proms.copy()

    # Округляем те значения, у которых порядок меньше максимального
    for i in range(len(peak_proms)):
        if peak_proms[i] != 0 and orders[i] < max_order:
            r_peak_proms[i] = np.round(peak_proms[i], -orders[i])

    orders = np.floor(np.log10(np.abs(r_peak_proms))).astype(int)

    unique_orders = np.unique(orders)
    unique_orders = unique_orders[unique_orders != max_order]

    lower_orders = unique_orders[unique_orders < max_order]

    if len(lower_orders) > 0:
        selected_mask = (orders == max_order) | (orders == lower_orders.max())
    else:
        selected_mask = (orders == max_order)

    #  Добавляем +1 балл в Points для выбранных по маске пиков
    Points[selected_mask] += 1

    # === ЭТАП 2. Оценка ширины пиков (peak_widths) от 0 до 1, где 1 - самый широкий, 0 - самый узкий ===
    idx_widths = np.argsort(-peak_width)        # sort desc
    sorted_widths = peak_width[idx_widths]
    n = len(sorted_widths)

    if n > 1:
        norm_widths = (sorted_widths - sorted_widths.min()) / (sorted_widths.max() - sorted_widths.min())
    else:
        norm_widths = np.array([1.0])

    for k in range(n):
        Points[idx_widths[k]] += norm_widths[k]

    # === ЭТАП 3. Соотношение peaks/peak_widths: 1 - наибольшее, 0 - наименьшее ===
    R = peak_heights / peak_width
    idx_R = np.argsort(-R)
    sorted_R = R[idx_R]
    nR = len(sorted_R)

    if nR > 1:
        norm_R = (sorted_R - sorted_R.min()) / (sorted_R.max() - sorted_R.min())
    else:
        norm_R = np.array([1.0])

    Points[idx_R] += norm_R

    # === ЭТАП 4. Пороговая фильтрация - если пик выше порога, то +1 ===
    selectedPeaks = corrected_signal[peak_indices]
    threshold = np.mean(selectedPeaks) / 3.0

    for i in range(len(peak_indices)):
        if corrected_signal[peak_indices[i]] > threshold:
            Points[i] += 1

    # === ЭТАП 5. Оценка "одиноких" пиков, если проходят под условия +-4 от максимума, значит это не локальные пики библиотеки, +1 ===
    for i in range(len(peak_indices)):
        left_val, max_val, right_val = dl_peaks(corrected_signal, peak_indices[i])

        # Проверка условия "одинокого пика"
        if (left_val < 0.9 * max_val and right_val < 0.9 * max_val) or orders[i] == max_order:
            Points[i] += 1

    # === ЭТАП 6: Сортировка first_reper_idx ===
    first_reper_idx = np.where(np.abs(peak_indices - standard_peaks[0]) <= 100)[0]

    vals = peak_indices[first_reper_idx]  # время выхода выбранных пиков
    idx_sort = np.argsort(vals)
    sorted_vals = vals[idx_sort]
    nVals = len(sorted_vals)

    if nVals > 1:
        # нормализация от 0 до 1
        norm_vals = (sorted_vals - sorted_vals.min()) / (sorted_vals.max() - sorted_vals.min())
        # инвертируем: min -> 1, max -> 0
        adj_vals = 0.5 - norm_vals
    else:
        adj_vals = np.array([0.5])  # если один пик

    for k in range(nVals):
        orig_idx = first_reper_idx[idx_sort[k]]
        Points[orig_idx] += adj_vals[k]

    # === Выбор пиков по Points ===
    max_point_val = int(np.floor(Points.max()))  # максимальное целое значение баллов
    peak_for_choose = np.where(np.floor(Points) == max_point_val)[0]  # пики с целым максимумом

    #  если их меньше 5 → понижаем порог до тех пор, пока не наберётся ≥5
    current_val = max_point_val - 1

    while len(peak_for_choose) < 5 and current_val >= 0:
        idx_extra = np.where(np.floor(Points) == current_val)[0]  # выбираем баллы только для тех пиков, которые мы отобрали
        peak_for_choose = np.unique(np.concatenate([peak_for_choose, idx_extra]))  # добавляем значения в массив и удаляем дубликаты
        current_val -= 1

    # === Определяем зоны первого и второго репера ===
    k = 100
    first_reper_mask = np.abs(peak_indices - standard_peaks[0]) <= k
    first_reper_idx = np.where(first_reper_mask)[0]

    #  Первые пики, лежащие в интервале первого репера
    pre_first_reper = peak_for_choose[np.isin(peak_for_choose, first_reper_idx)]
    #  Остальные пики — кандидаты на второй репер
    pre_second_reper = np.setdiff1d(peak_for_choose, pre_first_reper)

    #  === Если second_reper пуст, снижаем порог баллов (если было 3, как выше, то станет 2) ===
    while len(pre_second_reper) == 0 and current_val >= 0:

        #  уменьшаем разряд и добавляем новые пики
        idx_extra = np.where(np.floor(Points) == current_val)[0]
        peak_for_choose = np.unique(np.concatenate([peak_for_choose, idx_extra]))
        current_val -= 1

        #  пересчитываем зоны
        first_reper_idx = np.where(np.abs(peak_indices - standard_peaks[0]) <= k)[0]
        pre_first_reper = peak_for_choose[np.isin(peak_for_choose, first_reper_idx)]
        pre_second_reper = np.setdiff1d(peak_for_choose, pre_first_reper)

        #  === Если слишком много кандидатов — сужаем диапазон k ===
        if len(peak_for_choose) > 10:
            while len(peak_for_choose) > 10 and k > 50:
                k -= 10
                first_reper_idx = np.where(np.abs(peak_indices - standard_peaks[0]) <= k)[0]
                pre_first_reper = peak_for_choose[np.isin(peak_for_choose, first_reper_idx)]
                pre_second_reper = np.setdiff1d(peak_for_choose, pre_first_reper)

                #  если разделение удалось — выходим
                if len(pre_second_reper) > 0:
                    break

            # === После сужения диапазона удаляем пики с наименьшими баллами ===
            min_point_val = int(np.floor(Points[peak_for_choose]).min())
            idx_remove = np.where(np.floor(Points) == min_point_val)[0]

            # исключаем эти пики из выбора
            peak_for_choose = np.setdiff1d(peak_for_choose, idx_remove)

            # пересчитываем first/second reper заново после удаления
            first_reper_idx = np.where(np.abs(peak_indices - standard_peaks[0]) <= k)[0]
            pre_first_reper = peak_for_choose[np.isin(peak_for_choose, first_reper_idx)]
            pre_second_reper = np.setdiff1d(peak_for_choose, pre_first_reper)

    # Находим максимальный целый разряд среди них для очистки неопознанных пиков ниже
    pre_reper = np.unique(np.concatenate([pre_first_reper, pre_second_reper]))
    # Получаем баллы этих пиков
    pre_points = Points[pre_reper]
    max_order = int(np.floor(pre_points.max()))

    # Сортируем по убыванию Points для удобства
    sort_idx = np.argsort(-Points[pre_reper])
    pre_reper = pre_reper[sort_idx]

    # === Первый репер ===
    max_val = Points[pre_first_reper].max()
    first_reper_idx = pre_first_reper[Points[pre_first_reper] == max_val]

    if len(pre_first_reper) > 1:
        pre_first_reper = unrec_clean(pre_reper, max_order, Points, pre_first_reper)
        pre_first_reper = np.setdiff1d(pre_first_reper, first_reper_idx)
    else:
        pre_first_reper = np.array([], dtype=int)

    first_reper = peak_indices[first_reper_idx][0]

    # Находим границы от текущего selectedPeakLocations +-4
    left_idx = first_reper - 4
    right_idx = first_reper + 4

    # Ищем максимум между текующими границами - для выравнивания
    max_value = corrected_signal[left_idx:right_idx].max()
    first_reper = np.where(corrected_signal == max_value)[0][0]
    peak_indices[first_reper_idx] = first_reper

    # === Второй репер ===
    max_val = Points[pre_second_reper].max()
    second_reper_idx = pre_second_reper[Points[pre_second_reper] == max_val]

    if len(pre_second_reper) > 1:
        pre_second_reper = unrec_clean(pre_reper, max_order, Points, pre_second_reper)
        pre_second_reper = np.setdiff1d(pre_second_reper, second_reper_idx)
    else:
        pre_second_reper = np.array([], dtype=int)

    second_reper = peak_indices[second_reper_idx][0]

    #  Находим границы от текущего selectedPeakLocations +-4
    left_idx = second_reper - 4
    right_idx = second_reper + 4

    # Ищем максимум между текующими границами - для выравнивания
    max_value = corrected_signal[left_idx:right_idx].max()
    second_reper = np.where(corrected_signal == max_value)[0][0]
    peak_indices[second_reper_idx] = second_reper

    # Продолжение логики
    pre_unrecognized_peaks = np.unique(
        np.concatenate([peak_indices[pre_first_reper], peak_indices[pre_second_reper]])
    )

    reference_peaks = np.sort([first_reper, second_reper])

    selectedPeaks = corrected_signal[peak_indices]
    sd_Peaks = corrected_signal[reference_peaks]
    threshold = np.mean(sd_Peaks) / 4.0

    selectedPeaks = selectedPeaks[selectedPeaks >= threshold]
    selected_peaks = np.where(np.isin(corrected_signal, selectedPeaks))[0]

    return selected_peaks, pre_unrecognized_peaks, reference_peaks


def unrec_clean(pre_reper, max_order, Points, pre_find_reper):
    # Получаем баллы этих пиков
    pre_points = Points[pre_find_reper]

    # Оставляем только пики, у которых целый разряд Points равен max_order
    # или (max_order - 1)
    keep_mask = np.floor(pre_points) >= (max_order - 1)

    # Фильтруем только нужные пики
    pre_reper = pre_find_reper[keep_mask]

    return pre_reper


def dl_peaks(denoised_data, lonely_pks):
    # Текущий индекс
    peak_idx = lonely_pks

    # Находим границы от текущего selectedPeakLocations +-4
    left_idx = peak_idx - 4
    right_idx = peak_idx + 4

    left_idx = max(left_idx, 0)
    right_idx = min(right_idx, len(denoised_data) - 1)

    # Ищем максимум между текующими границами
    window = denoised_data[left_idx:right_idx + 1]
    max_value = np.max(window)

    peak_indices = np.where(denoised_data == max_value)[0]

    peak_idx = peak_indices[0]

    #  Находим границы от текущего максимального значения +-4
    left_idx = max(peak_idx - 4, 0)
    right_idx = min(peak_idx + 4, len(denoised_data) - 1)

    # Значения слева и справа
    left_value = denoised_data[left_idx]
    right_value = denoised_data[right_idx]

    return left_value, max_value, right_value
