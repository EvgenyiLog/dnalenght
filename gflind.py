from __future__ import annotations

import warnings
import numpy as np
from typing import Tuple, Optional
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import gaussian_filter1d

def _find_max_peak(denoised_data, peak_locations, reper_idx):
    """Уточняет позицию пика: ищет максимум ±4 отсчёта вокруг peak_locations[reper_idx]."""
    reper = int(peak_locations[reper_idx])
    lo = max(0, reper - 4)
    hi = min(len(denoised_data) - 1, reper + 4)
    seg = denoised_data[lo:hi + 1]
    return lo + int(np.argmax(seg))


def _dl_peaks(denoised_data, pk_idx):
    """
    Возвращает (left_value, max_value, right_value) для «одинокого» пика.
    Смотрит ±4 отсчёта вокруг pk_idx.
    """
    pk_idx = int(pk_idx)
    lo = max(0, pk_idx - 4)
    hi = min(len(denoised_data) - 1, pk_idx + 4)
    seg = denoised_data[lo:hi + 1]
    max_val = float(np.max(seg))

    local_max = lo + int(np.argmax(seg))
    left_idx = max(0, local_max - 4)
    right_idx = min(len(denoised_data) - 1, local_max + 4)

    return float(denoised_data[left_idx]), max_val, float(denoised_data[right_idx])


def _hid_fun(denoised_data, start_index, end_index, px,
             hidden_lib_areas, max_lib_value):
    """
    Вычисление «скрытых» позиций и площадей внутри ГБ.
    Аналог MATLAB Hid_fun: находит медианные точки от start до max и от max до end,
    строит массив Hidden_LibPeakLocations и считает площади между ними.
    """
    start_index = int(start_index)
    end_index = int(end_index)
    mlv = int(max_lib_value) if np.isscalar(max_lib_value) else int(max_lib_value[0])

    # MATLAB: median(start_index:maxLibValue) = (start_index + maxLibValue) / 2
    med_before = (start_index + mlv) / 2.0
    med_after = (mlv + end_index) / 2.0

    rounded = np.round([med_before, med_after]).astype(int)
    # Hidden_LibPeakLocations — массив от rounded[0] до rounded[1]
    hidden_locs = np.arange(rounded[0], rounded[1] + 1)

    # Hidden_final_Lib_local_minimums = [start_index, hidden_locs...]
    hidden_mins = np.concatenate([[start_index], hidden_locs])

    # Считаем площади для каждого отрезка hidden_mins[i]:hidden_mins[i+1]
    new_areas = []
    n_dd = len(denoised_data)
    for i in range(len(hidden_mins) - 1):
        a = int(hidden_mins[i])
        b = int(hidden_mins[i + 1])
        if b > a and b < n_dd and a >= 0:
            new_areas.append(float(np.trapezoid(denoised_data[a:b + 1])))
        else:
            new_areas.append(0.0)

    # Объединяем с ранее накопленными площадями
    prev_areas = list(hidden_lib_areas) if len(hidden_lib_areas) else []
    all_areas = prev_areas + new_areas

    return hidden_locs, np.array(all_areas, dtype=np.float64), hidden_mins


def _unrec_clean(pre_reper, max_order, Points, pre_find_reper):
    """Оставляет только пики, чей целый разряд Points >= max_order - 1."""
    pre_points = Points[pre_find_reper]
    mask = np.floor(pre_points) >= (max_order - 1)
    return pre_find_reper[mask]

def _make_empty_result():
    """Возвращает словарь с пустыми значениями (для раннего выхода)."""
    ea = np.array([])
    ei = np.array([], dtype=int)
    return {
        't_main': ea, 'denoised_data': ea,
        'st_peaks': ea, 'st_length': ei,
        't_unrecognized_peaks': ea, 'unrecognized_peaks': ei,
        'lib_length': ea, 'LibPeakLocations': ei,
        't_final_locations': ea, 'final_Lib_local_minimums': ei,
        'hpx': ei, 'unr': ei, 'stp': ea,
        'all_areas': ea, 'all_peaksCorr': ea,
        'all_peaks': ea, 'all_areasConc': ea,
        'molarity': ea,
        'maxLibPeak': None, 'maxLibValue': None,
        'totalLibArea': 0.0, 'totalLibConc': 0.0, 'totalLibMolarity': 0.0,
        'x_fill': ea, 'y_fill': ea,
        'x_Lib_fill': ea, 'y_Lib_fill': ea,
    }

# ---------------------------------------------------------------------------
# findpeaks_matlab — поиск пиков (аналог MATLAB findpeaks)
# ---------------------------------------------------------------------------
from scipy.signal import find_peaks, peak_prominences, peak_widths as _peak_widths
def findpeaks_matlab(data, min_peak_height=None, min_peak_distance=None):
    """
    Поиск пиков по аналогии с MATLAB findpeaks.

    Возвращает
    ----------
    values, locations (int), widths, prominences
    """
    data = np.asarray(data, dtype=np.float64)

    kw = {}
    if min_peak_height is not None:
        kw['height'] = min_peak_height
    if min_peak_distance is not None:
        kw['distance'] = int(min_peak_distance)

    locations, _ = find_peaks(data, **kw)

    if len(locations) == 0:
        _e = np.empty(0, dtype=np.float64)
        return _e, np.empty(0, dtype=np.intp), _e, _e

    values = data[locations]
    prom = peak_prominences(data, locations)[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wid = _peak_widths(data, locations, rel_height=0.5)[0]
    return values, locations, wid, prom


def glfind(
    data: np.ndarray,
    peaks:np.ndarray,
    LIZ: np.ndarray,
    CONC: np.ndarray,
    override_repers: Optional [np.ndarray] = None
    ) -> Dict[str, Any]:
   
    """
    Полный анализ геномной библиотеки.

    Параметры
    ---------
    data : 1-D array — сырые данные геномной библиотеки
    peak : 1-D int array — позиции пиков стандарта длин (из SDFind1_2)
    LIZ  : 1-D array — длины фрагментов стандарта (пн)
    CONC : 1-D array — концентрации стандарта (нг/мкл)
    override_repers : 1-D int array или None — если задан, используется
        вместо алгоритмически найденных реперов (индексы в denoised_data)

    Возвращает
    ----------
    dict с ключами:
        t_main, denoised_data, st_peaks, st_length,
        t_unrecognized_peaks, unrecognized_peaks,
        lib_length, LibPeakLocations, t_final_locations, final_Lib_local_minimums,
        hpx, unr, stp, mainCorr,
        all_areas, all_peaksCorr, all_peaks, all_areasConc, molarity,
        maxLibPeak, maxLibValue, totalLibArea, totalLibConc, totalLibMolarity,
        x_fill, y_fill, x_Lib_fill, y_Lib_fill
    """
    data = np.asarray(data, dtype=float).flatten()
    peak = np.asarray(peaks, dtype=int).flatten()
    LIZ = np.asarray(LIZ, dtype=float).flatten()
    CONC = np.asarray(CONC, dtype=float).flatten()

    _empty = _make_empty_result()
    denoised_data=data
    x = np.arange(len(denoised_data), dtype=float)
    denoised_data[denoised_data < 0] = 0

    # ================================================================
    # 2. Фильтрация + нахождение пиков на 2-й производной
    # ================================================================
    filt = savgol_filter(denoised_data, 5, 1)
    d1 = np.diff(filt)
    filt_d1 = savgol_filter(d1, 5, 1)
    d2 = np.diff(filt_d1)
    filt_d2 = savgol_filter(d2, 5, 1)

    flipped = -filt_d2.copy()
    flipped[flipped < 0] = 0

    peaks_v, peak_locs, peak_w, peak_p = findpeaks_matlab(flipped)

    if len(peak_locs) == 0:
        return _empty

    # Отсеиваем крайние пики
    valid = (peak_locs > 15) & (peak_locs < (len(x) - 10))
    peak_locs = peak_locs[valid]
    peaks_v = peaks_v[valid]
    peak_w = peak_w[valid]
    peak_p = peak_p[valid]

    if len(peak_locs) == 0:
        return _empty

    # ================================================================
    # Система баллов Points
    # ================================================================
    Points = np.zeros(len(peak_p), dtype=float)

    # --- ЭТАП 1: разряды ---
    with np.errstate(divide='ignore', invalid='ignore'):
        orders = np.floor(np.log10(np.abs(peak_p)))
    orders[~np.isfinite(orders)] = 0
    max_order_val = np.max(orders)

    r_peak_p = peak_p.copy()
    for i in range(len(peak_p)):
        if peak_p[i] != 0 and orders[i] < max_order_val:
            r_peak_p[i] = np.round(peak_p[i], int(-orders[i]))

    with np.errstate(divide='ignore', invalid='ignore'):
        orders = np.floor(np.log10(np.abs(r_peak_p)))
    orders[~np.isfinite(orders)] = 0

    unique_ord = np.unique(orders)
    unique_ord = unique_ord[unique_ord != max_order_val]
    lower_ord = unique_ord[unique_ord < max_order_val]
    nearest_order = np.max(lower_ord) if len(lower_ord) else None

    if nearest_order is not None:
        sel_mask = (orders == max_order_val) | (orders == nearest_order)
    else:
        sel_mask = (orders == max_order_val)

    Points[sel_mask] += 1

    # --- ЭТАП 2: ширина пиков ---
    idx_w = np.argsort(peak_w)[::-1]
    sw = peak_w[idx_w]
    n = len(sw)
    if n > 1:
        rng = sw.max() - sw.min()
        norm_w = (sw - sw.min()) / rng if rng > 0 else np.ones(n)
    else:
        norm_w = np.array([1.0])
    for k in range(n):
        Points[idx_w[k]] += norm_w[k]

    # --- ЭТАП 3: соотношение peaks/widths ---
    R = peaks_v / np.where(peak_w == 0, 1e-10, peak_w)
    idx_r = np.argsort(R)[::-1]
    sr = R[idx_r]
    nr = len(sr)
    if nr > 1:
        rng = sr.max() - sr.min()
        norm_r = (sr - sr.min()) / rng if rng > 0 else np.ones(nr)
    else:
        norm_r = np.array([1.0])
    Points[idx_r] += norm_r

    # --- ЭТАП 4: пороговая фильтрация (векторизовано) ---
    sel_pks = denoised_data[peak_locs]
    thr = np.mean(sel_pks) / 3.0
    Points[sel_pks > thr] += 1

    # --- ЭТАП 5: «одинокие» пики ---
    for i in range(len(peak_locs)):
        lv, mv, rv = _dl_peaks(denoised_data, peak_locs[i])
        if (lv < 0.9 * mv and rv < 0.9 * mv) or orders[i] == max_order_val:
            Points[i] += 1

    if peak is None or len(peak) == 0:
        print("Ошибка GLFind: массив peak пуст или не определён!")
        return {}  # Или верните None, в зависимости от ожидаемого типа glfinddict

    # --- ЭТАП 6: сортировка по близости к первому реперу ---
    first_reper_idx_arr = np.where(np.abs(peak_locs.astype(float) - peak[0]) <= 100)[0]
    if len(first_reper_idx_arr):
        vals = peak_locs[first_reper_idx_arr].astype(float)
        idx_sort = np.argsort(vals)
        sv = vals[idx_sort]
        nv = len(sv)
        if nv > 1:
            rng = sv.max() - sv.min()
            nv_ = (sv - sv.min()) / rng if rng > 0 else np.zeros(nv)
            adj = 0.5 - nv_
        else:
            adj = np.array([0.5])
        for k in range(nv):
            Points[first_reper_idx_arr[idx_sort[k]]] += adj[k]

    # ================================================================
    # Выбор пиков по баллам
    # ================================================================
    led_check = np.round(Points).astype(int)
    max_v = int(np.max(led_check))
    count_max = int(np.sum(led_check == max_v))
    count_max2 = 0
    if count_max == 1:
        tmp = led_check[led_check != max_v]
        if len(tmp):
            mv2 = int(np.max(tmp))
            count_max2 = int(np.sum(led_check == mv2))
    count_max += count_max2

    led_idx = np.array([], dtype=int)
    phix_idx = np.array([], dtype=int)
    n_rep = len(peak)

    pre_first_reper = np.array([], dtype=int)
    pre_second_reper = np.array([], dtype=int)

    # ---------- ветвь 1: count_max > 5 и <= n_rep ----------
    if count_max > 5 and count_max <= n_rep:
        sorted_idx = np.argsort(Points)[::-1]
        best_idx = np.sort(sorted_idx[:n_rep])
        best_pl = peak_locs[best_idx]
        idx_min = int(np.argmin(best_pl))
        idx_max = int(np.argmax(best_pl))
        pre_first_reper = np.array([best_idx[idx_min]])
        pre_second_reper = np.array([best_idx[idx_max]])

        for k in range(len(best_idx)):
            if k != idx_min and k != idx_max:
                if (peak_locs[best_idx[k]] > peak_locs[pre_first_reper[0]]
                        and peak_locs[best_idx[k]] < peak_locs[pre_second_reper[0]]):
                    led_idx = np.append(led_idx, best_idx[k])

    # ---------- ветвь 2: count_max > 2 и <= 5 ----------
    elif count_max > 2 and count_max <= 5:
        n_rep_local = 3
        sorted_idx = np.argsort(Points)[::-1]
        sorted_prom_idx = np.argsort(peak_p)[::-1]
        best_idx = np.sort(sorted_idx[:n_rep_local])
        best_proms_idx = sorted_prom_idx[:n_rep_local]

        pre_first_reper = np.array([best_idx[0]])
        pre_second_reper = np.array([best_idx[-1]])

        for k in range(1, len(best_idx) - 1):
            if (peak_locs[best_idx[k]] > peak_locs[pre_first_reper[0]]
                    and peak_locs[best_idx[k]] < peak_locs[pre_second_reper[0]]):
                phix_idx = np.append(phix_idx, best_idx[k])

        # Проверка: есть ли ГБ между реперами
        left_bound = peak_locs[pre_first_reper[0]] + 20
        right_bound = peak_locs[pre_second_reper[0]] - 20

        # Вместо построения set диапазона — используем numpy маску
        excl_mask = np.zeros(right_bound - left_bound + 1, dtype=bool) if right_bound >= left_bound else np.array([], dtype=bool)
        if len(excl_mask):
            for ii in phix_idx:
                lo_e = max(0, peak_locs[ii] - 30 - left_bound)
                hi_e = min(len(excl_mask), peak_locs[ii] + 31 - left_bound)
                excl_mask[lo_e:hi_e] = True

            all_idx = np.arange(left_bound, right_bound + 1)
            valid_idx = all_idx[~excl_mask]
        else:
            valid_idx = np.array([], dtype=int)
        if len(valid_idx) > 0:
            check_lib = denoised_data[valid_idx]
            thr2 = np.mean([denoised_data[peak_locs[pre_first_reper[0]]],
                            denoised_data[peak_locs[pre_second_reper[0]]]]) / 2
            check_status = 1 if np.sum(check_lib > thr2) > 5 else 0
        else:
            check_status = 0

        if check_status == 1:
            pre_first_reper = np.array([best_idx[0]])
            pre_second_reper = np.array([best_idx[1]])
            phix_idx = np.array([], dtype=int)

        phix_proms = np.sort(peak_p[np.concatenate([pre_first_reper, phix_idx, pre_second_reper])])[::-1] if len(phix_idx) else np.array([])
        best_proms = peak_p[best_proms_idx]

        if (not np.array_equal(phix_proms, best_proms)) or check_status == 1:
            phix_idx = np.array([], dtype=int)
            pre_first_reper, pre_second_reper = _select_repers_by_points(
                Points, peak_locs, peak, orders, max_order_val)

    # ---------- ветвь 3: иначе ----------
    else:
        pre_first_reper, pre_second_reper = _select_repers_by_points(
            Points, peak_locs, peak, orders, max_order_val)

    # === Проверка override реперов ===
    _use_override = override_repers is not None and len(override_repers) >= 2

    if not _use_override and (len(pre_first_reper) == 0 or len(pre_second_reper) == 0):
        # Реперы не найдены — вернуть результат с denoised_data для ручного редактирования
        partial = _make_empty_result()
        partial['denoised_data'] = denoised_data
        partial['t_main'] = np.arange(len(denoised_data), dtype=float)
        partial['_repers_not_found'] = True
        return partial

    # Dummy значения чтобы код рефайнмента не падал при override
    if _use_override:
        if len(pre_first_reper) == 0:
            pre_first_reper = np.array([0], dtype=int)
        if len(pre_second_reper) == 0:
            pre_second_reper = np.array([max(1, len(peak_locs) - 1)], dtype=int)

    # ================================================================
    # Определение максимального разряда среди кандидатов
    # ================================================================
    pre_reper = np.unique(np.concatenate([pre_first_reper, pre_second_reper]))
    pre_points = Points[pre_reper]
    max_order_pts = int(np.floor(np.max(pre_points)))

    sort_idx = np.argsort(Points[pre_reper])[::-1]
    pre_reper = pre_reper[sort_idx]

    # --- Первый репер ---
    mv1 = np.max(Points[pre_first_reper])
    first_reper_idx_final = pre_first_reper[Points[pre_first_reper] == mv1]
    if len(first_reper_idx_final) > 1:
        first_reper_idx_final = first_reper_idx_final[:1]

    if len(pre_first_reper) > 1:
        cleaned = _unrec_clean(pre_reper, max_order_pts, Points, pre_first_reper)
        pre_first_reper = np.setdiff1d(cleaned, first_reper_idx_final)
    else:
        pre_first_reper = np.array([], dtype=int)

    first_reper = _find_max_peak(denoised_data, peak_locs, first_reper_idx_final[0])
    peak_locs[first_reper_idx_final[0]] = first_reper

    # --- Второй репер ---
    mv2 = np.max(Points[pre_second_reper])
    second_reper_idx_final = pre_second_reper[Points[pre_second_reper] == mv2]
    if len(second_reper_idx_final) > 1:
        second_reper_idx_final = second_reper_idx_final[:1]

    if len(pre_second_reper) > 1:
        cleaned2 = _unrec_clean(pre_reper, max_order_pts, Points, pre_second_reper)
        pre_second_reper = np.setdiff1d(cleaned2, second_reper_idx_final)
    else:
        pre_second_reper = np.array([], dtype=int)

    second_reper = _find_max_peak(denoised_data, peak_locs, second_reper_idx_final[0])
    peak_locs[second_reper_idx_final[0]] = second_reper

    # ================================================================
    # Формирование массивов
    # ================================================================
    pre_unrecognized_peaks = np.unique(np.concatenate([
        peak_locs[pre_first_reper] if len(pre_first_reper) else np.array([], dtype=int),
        peak_locs[pre_second_reper] if len(pre_second_reper) else np.array([], dtype=int),
    ]))

    st_length = np.sort([first_reper, second_reper])

    sel_pks2 = denoised_data[peak_locs]
    sd_pks = denoised_data[st_length]
    thr2 = np.mean(sd_pks) / 4
    sel_pks2 = sel_pks2[sel_pks2 >= thr2]

    # Обработка led_idx / phix_idx
    if len(led_idx):
        _extra_lr = [_find_max_peak(denoised_data, peak_locs, led_idx[i])
                     for i in range(len(led_idx))]
        pre_unrecognized_peaks = np.sort(np.unique(
            np.concatenate([pre_unrecognized_peaks, _extra_lr])))
        selectedPeakLocations = pre_unrecognized_peaks.copy()

    elif len(phix_idx):
        tmp_list = list(pre_unrecognized_peaks)
        for i in range(len(phix_idx)):
            pr = _find_max_peak(denoised_data, peak_locs, phix_idx[i])
            tmp_list.append(pr)
        selectedPeakLocations = np.sort(np.unique(np.array(tmp_list, dtype=int)))

    else:
        if len(sel_pks2):
            # Вместо np.isin по float — ищем индексы пиков выше порога
            mask_above = denoised_data[peak_locs] >= thr2
            selectedPeakLocations = peak_locs[mask_above]
        else:
            selectedPeakLocations = np.array([], dtype=int)

    # === Применение override реперов ===
    if _use_override:
        st_length = np.sort(np.asarray(override_repers, dtype=int))
        st_length = np.clip(st_length, 0, len(denoised_data) - 1)
        pre_unrecognized_peaks = np.array([], dtype=int)
        sd_pks = denoised_data[st_length]
        thr2 = np.mean(sd_pks) / 4 if len(sd_pks) > 0 else 0
        mask_above = denoised_data[peak_locs] >= thr2
        selectedPeakLocations = peak_locs[mask_above]

    if len(selectedPeakLocations) == 0:
        return _empty

    # ================================================================
    # 3. Нахождение минимумов
    # ================================================================
    flip_dd = -denoised_data
    _, min_pk_locs, _, _ = findpeaks_matlab(flip_dd, min_peak_distance=8)

    if len(min_pk_locs) > 0:
        pk_thr = 0.6 * np.mean(flip_dd)
        below_thr = flip_dd[min_pk_locs] < pk_thr
        all_local_minimums = min_pk_locs[below_thr]
        all_local_minimums = np.setdiff1d(all_local_minimums, selectedPeakLocations)
        min_pk_locs = min_pk_locs[~below_thr]
    else:
        all_local_minimums = np.array([], dtype=int)
        min_pk_locs = np.array([], dtype=int)

    crossings2 = np.where(np.diff((denoised_data > 0).astype(int)))[0]
    complete_locs = np.union1d(min_pk_locs, crossings2).astype(int)

    # ================================================================
    # 4. Калибровка
    # ================================================================
    st_peaks = np.array([peak[0], peak[-1]])
    CONC_pair = np.array([CONC[0], CONC[-1]])
    inLIZ = LIZ.copy()

    SDC = np.polyfit(peak, inLIZ, 5)
    SDC2 = np.polyfit(inLIZ, peak, 5)

    if len(st_length) != 2:
        return _empty

    # Проверка минимумов вблизи реперов
    min_st_length = []
    for i in range(len(st_length)):
        cv = st_length[i]
        left_c = complete_locs[complete_locs < cv]
        right_c = complete_locs[complete_locs > cv]
        closest = []
        if len(left_c):
            closest.append(int(np.max(left_c)))
        if len(right_c):
            closest.append(int(np.min(right_c)))
        if closest and any(abs(c - cv) > 10 for c in closest):
            min_st_length.extend([cv - 7, cv + 7])

    if min_st_length:
        complete_locs = np.unique(np.sort(
            np.concatenate([complete_locs, np.array(min_st_length, dtype=int)])))

    # Удаление пересекающихся минимумов рядом с реперами
    for i in range(len(st_length)):
        cv = int(st_length[i])
        nearby = pre_unrecognized_peaks[
            (pre_unrecognized_peaks >= cv - 10) & (pre_unrecognized_peaks <= cv + 10)]
        for tp in nearby:
            rs = min(cv, int(tp))
            re = max(cv, int(tp))
            if rs >= 0 and re < len(denoised_data):
                seg = denoised_data[rs:re + 1]
                mi = rs + int(np.argmin(seg))
                complete_locs = complete_locs[(complete_locs < rs) | (complete_locs > re)]
                complete_locs = np.unique(np.sort(np.append(complete_locs, mi)))

    # Фильтрация пиков за пределами реперов
    selectedPeakLocations = selectedPeakLocations[
        (selectedPeakLocations >= st_length[0]) & (selectedPeakLocations <= st_length[-1])]
    selectedPeakLocations = np.unique(np.sort(
        np.concatenate([[st_length[0]], selectedPeakLocations, [st_length[-1]]])))

    check_selectPeaks = np.setdiff1d(selectedPeakLocations,
                                      np.concatenate([pre_unrecognized_peaks, st_length]))

    # --- Линейная калибровка по реперам ---
    px = np.polyfit(st_length, st_peaks, 1)
    t = np.arange(len(denoised_data), dtype=float)
    t_main = np.polyval(px, t)

    # ================================================================
    # 5. Основной цикл — разбиение пиков по классам
    # ================================================================
    LibPeakLocations = []
    Hidden_LibPeakLocations = np.array([], dtype=int)
    maxLibValue = None
    lib_peaksCorr = np.array([])

    unrecognized_peaks = np.array([], dtype=int)
    rest_peaks = np.array([], dtype=int)

    st_areas = []
    lib_areas = np.array([], dtype=np.float64)
    Hidden_lib_areas = np.array([], dtype=np.float64)
    rest_peaks_areas = np.array([], dtype=np.float64)

    lib_one_area = np.array([], dtype=np.float64)
    lib_one_areaConc = np.array([], dtype=np.float64)

    lib_molarity = np.array([], dtype=np.float64)

    final_Lib_local_minimums = []
    Hidden_final_Lib_local_minimums = np.array([], dtype=int)

    x_fill_1 = np.array([], dtype=np.float64)
    x_Lib_fill_1 = np.array([], dtype=np.float64)
    y_fill = np.array([], dtype=np.float64)
    y_Lib_fill = np.array([], dtype=np.float64)

    lib_check = 1
    n_dd = len(denoised_data)
    dd_indices = np.arange(n_dd, dtype=np.float64)  # предвычисляем для np.interp

    # Предвычисляем set для быстрого lookup
    pre_unrec_set = set(pre_unrecognized_peaks.tolist())
    st_length_set = set(st_length.tolist())

    for i in range(len(complete_locs) - 1):
        c_lo = complete_locs[i]
        c_hi = complete_locs[i + 1]
        between = (selectedPeakLocations >= c_lo) & \
                  (selectedPeakLocations <= c_hi)
        n_between = int(np.sum(between))

        if 0 < n_between < 4:
            x_start = int(c_lo)
            x_end = int(c_hi)

            # Проверка через set вместо np.isin + np.arange
            has_unrec = any(x_start <= v <= x_end for v in pre_unrec_set)
            has_st = any(x_start <= v <= x_end for v in st_length_set)

            if has_unrec:
                exact = pre_unrecognized_peaks[
                    (pre_unrecognized_peaks >= x_start) &
                    (pre_unrecognized_peaks <= x_end)]
                unrecognized_peaks = np.concatenate(
                    [unrecognized_peaks, exact.astype(int)])
                x_fill_1 = np.linspace(x_start, x_end, 100)
                y_fill = np.interp(x_fill_1, dd_indices, denoised_data)

            elif has_st:
                area_val = float(np.trapezoid(denoised_data[x_start:x_end + 1]))
                st_areas.append(area_val)

        elif n_between >= 4 and lib_check == 1:
            start_index = int(complete_locs[i])
            end_index = int(complete_locs[i + 1])

            seg = denoised_data[start_index:end_index + 1]
            max_val_lib = float(np.max(seg))
            max_idx_lib = start_index + int(np.argmax(seg))
            maxLibValue = max_idx_lib

            maxLibValueCorr = np.polyval(SDC, max_idx_lib)
            lb = maxLibValueCorr - 200
            ub = maxLibValueCorr + 200
            lb_idx = np.polyval(SDC2, lb)
            ub_idx = np.polyval(SDC2, ub)

            if lb_idx > start_index or ub_idx < end_index:
                x_Lib_fill_1 = np.linspace(start_index, end_index, 100)
                y_Lib_fill = np.interp(x_Lib_fill_1, dd_indices, denoised_data)

            # Локальные пики ГБ
            between_pks = selectedPeakLocations[between]
            final_Lib_local_minimums = list(between_pks) + [int(complete_locs[i + 1])]

            lib_pks = []
            for j in range(len(final_Lib_local_minimums) - 1):
                xs = int(final_Lib_local_minimums[j])
                xe = int(final_Lib_local_minimums[j + 1])
                if xs < xe and xe < len(denoised_data):
                    seg2 = denoised_data[xs:xe + 1]
                    mx = xs + int(np.argmax(seg2))
                    lib_pks.append(mx)

            LibPeakLocations = sorted(set(lib_pks))

            # Проверка: фаикс / слишком низкие пики
            if LibPeakLocations:
                diff_vals = np.abs(denoised_data[LibPeakLocations] - max_val_lib)
                thr20 = 0.2 * max_val_lib
                check_lp = diff_vals > thr20
                diff_count = len(LibPeakLocations) - int(np.sum(check_lp))

                one_pks_count = 0
                for lp in LibPeakLocations:
                    lv, mv, rv = _dl_peaks(denoised_data, lp)
                    if lv < 0.92 * mv and rv < 0.92 * mv:
                        one_pks_count += 1

                if diff_count < 0.3 * len(LibPeakLocations) and one_pks_count == 1:
                    LibPeakLocations = []
                else:
                    Hidden_LibPeakLocations, Hidden_lib_areas, Hidden_final_Lib_local_minimums = \
                        _hid_fun(denoised_data, start_index, end_index, px,
                                 Hidden_lib_areas, maxLibValue)

            lib_check += 1

    # ================================================================
    # Расчёт концентраций, площадей, молярностей
    # ================================================================
    mainCorr = np.polyval(SDC, t_main)
    st_peaksCorr = np.polyval(SDC, st_peaks)

    st_areas_arr = np.array(st_areas) if st_areas else np.array([0.0, 0.0])
    if len(st_areas_arr) >= 2:
        st_areas_pair = np.array([st_areas_arr[0], st_areas_arr[-1]])
    else:
        st_areas_pair = np.array([0.0, 0.0])

    safe_corr = np.where(st_peaksCorr == 0, 1e-10, st_peaksCorr)
    led_one_area = st_areas_pair / (safe_corr / 100)
    # Защита от деления на 0 при polyfit
    if np.all(led_one_area == 0) or np.all(led_one_area == led_one_area[0]):
        a_poly = np.array([0.0, 0.0])  # fallback
    else:
        a_poly = np.polyfit(led_one_area, CONC_pair, 1)
    st_molarity = ((CONC_pair * 1e-3) / (649 * st_peaksCorr)) * 1e9

    # --- Если ГБ гладкая / не найдена ---
    if len(Hidden_LibPeakLocations) == 0 and len(check_selectPeaks):
        rest_peaks_arr = selectedPeakLocations.copy()
        # Быстрая фильтрация вместо цикла с isin
        mask_rp = np.ones(len(rest_peaks_arr), dtype=bool)
        for s in st_length:
            mask_rp &= ~((rest_peaks_arr >= s - 7) & (rest_peaks_arr <= s + 7))
        rest_peaks_arr = rest_peaks_arr[mask_rp]

        rest_locs = np.sort(np.union1d(complete_locs, all_local_minimums))
        rest_areas_list = []

        i = 0
        while i < len(rest_locs) - 1:
            btw = (rest_peaks_arr >= rest_locs[i]) & (rest_peaks_arr <= rest_locs[i + 1])
            nb = int(np.sum(btw))
            pks_btw = rest_peaks_arr[btw]

            if nb > 1 and len(pks_btw) >= 2:
                new_peak = int(np.mean(pks_btw[:2]))
                rest_locs = np.unique(np.sort(np.append(rest_locs, new_peak)))
                i -= 1
            elif nb == 1:
                xs = int(rest_locs[i])
                xe = int(rest_locs[i + 1])
                if 0 <= xs and xe < n_dd:
                    rest_areas_list.append(float(np.trapezoid(denoised_data[xs:xe + 1])))
            i += 1

        rest_peaks_areas = np.array(rest_areas_list, dtype=np.float64)
        if len(rest_peaks_areas) and len(rest_peaks_arr):
            mx_idx = int(np.argmax(rest_peaks_areas))
            if mx_idx < len(rest_peaks_arr):
                LibPeakLocations = [int(rest_peaks_arr[mx_idx])]
                mv_val = int(rest_peaks_arr[mx_idx])
                mask_keep = np.abs(rest_peaks_arr - mv_val) > 4
                rest_peaks_arr = rest_peaks_arr[mask_keep]
                rest_peaks_areas = rest_peaks_areas[mask_keep] if mx_idx < len(rest_peaks_areas) else np.array([])
                unrecognized_peaks = rest_peaks_arr

                si = int(np.max(rest_locs[rest_locs < LibPeakLocations[0]])) if np.any(rest_locs < LibPeakLocations[0]) else 0
                ei = int(np.min(rest_locs[rest_locs > LibPeakLocations[0]])) if np.any(rest_locs > LibPeakLocations[0]) else len(denoised_data) - 1

                Hidden_LibPeakLocations, Hidden_lib_areas, Hidden_final_Lib_local_minimums = \
                    _hid_fun(denoised_data, si, ei, px, np.array([]), LibPeakLocations[0])
                final_Lib_local_minimums = [int(Hidden_final_Lib_local_minimums[0]),
                                            int(Hidden_final_Lib_local_minimums[-1])]

    # --- Пересчёт скрытых пиков ---
    if len(Hidden_LibPeakLocations):
        Hid_lib_length = np.polyval(px, Hidden_LibPeakLocations.astype(float))
        Hid_lib_peaksCorr = np.polyval(SDC, Hid_lib_length)
        safe_hid_corr = np.where(np.abs(Hid_lib_peaksCorr) < 1e-10, 1e-10, Hid_lib_peaksCorr)
        if len(Hidden_lib_areas) > 0:
            # Обеспечим совпадение размеров
            n_common = min(len(Hidden_lib_areas), len(safe_hid_corr))
            Hid_one_area = Hidden_lib_areas[:n_common] / (safe_hid_corr[:n_common] / 100)
            Hid_one_areaConc = np.polyval(a_poly, Hid_one_area)
            Hid_molarity = ((Hid_one_areaConc * 1e-3) / (649 * safe_hid_corr[:n_common])) * 1e9
        else:
            Hid_one_area = np.array([])
            Hid_one_areaConc = np.array([])
            Hid_molarity = np.array([])
    else:
        Hid_lib_peaksCorr = np.array([])
        Hid_one_area = np.array([])
        Hid_one_areaConc = np.array([])
        Hid_molarity = np.array([])

    LibPeakLocations = np.array(LibPeakLocations, dtype=int)
    final_Lib_local_minimums_arr = np.array(final_Lib_local_minimums, dtype=int) \
        if final_Lib_local_minimums else np.array([], dtype=int)

    # --- Расчёт ГБ ---
    lib_length = np.array([])
    lib_areas_out = np.array([])
    lib_one_area_out = np.array([])
    lib_one_areaConc_out = np.array([])
    lib_molarity_out = np.array([])

    if len(rest_peaks_areas) == 0 and len(check_selectPeaks) > 0 and len(LibPeakLocations) > 0 and len(Hidden_LibPeakLocations) > 0:
        # ГБ с локальными пиками
        # Проверяем, лежат ли LibPeakLocations в пределах Hidden_LibPeakLocations
        lp_in_hid = LibPeakLocations[
            (LibPeakLocations >= Hidden_LibPeakLocations[0]) &
            (LibPeakLocations <= Hidden_LibPeakLocations[-1])]

        # Если есть пики за границами — добавляем границы
        check_outside = LibPeakLocations[
            (LibPeakLocations < Hidden_LibPeakLocations[0]) |
            (LibPeakLocations > Hidden_LibPeakLocations[-1])]
        if len(check_outside) > 0:
            lp_in_hid = np.unique(np.sort(np.concatenate([
                [Hidden_LibPeakLocations[0]], lp_in_hid, [Hidden_LibPeakLocations[-1]]])))

        LibPeakLocations = np.unique(np.sort(lp_in_hid))

        lib_length = np.polyval(px, LibPeakLocations.astype(float))
        lib_peaksCorr = np.polyval(SDC, lib_length)

        # Суммирование скрытых площадей по фрагментам ГБ
        # Находим индексы LibPeakLocations в Hidden_LibPeakLocations
        if len(LibPeakLocations) > 0 and len(Hidden_LibPeakLocations) > 0:
            indices = []
            for lp in LibPeakLocations:
                idx_arr = np.where(Hidden_LibPeakLocations == lp)[0]
                if len(idx_arr) > 0:
                    indices.append(idx_arr[0])

            if len(indices) > 0:
                prev = 0
                _la, _loa, _lac, _lm = [], [], [], []
                for idx_val in indices:
                    end_idx = min(idx_val + 1, len(Hidden_lib_areas))
                    _la.append(float(np.sum(Hidden_lib_areas[prev:end_idx])))
                    _loa.append(float(np.sum(Hid_one_area[prev:end_idx])) if end_idx <= len(Hid_one_area) else 0.0)
                    _lac.append(float(np.sum(Hid_one_areaConc[prev:end_idx])) if end_idx <= len(Hid_one_areaConc) else 0.0)
                    _lm.append(float(np.sum(Hid_molarity[prev:end_idx])) if end_idx <= len(Hid_molarity) else 0.0)
                    prev = end_idx
                lib_areas_out = np.array(_la, dtype=np.float64)
                lib_one_area_out = np.array(_loa, dtype=np.float64)
                lib_one_areaConc_out = np.array(_lac, dtype=np.float64)
                lib_molarity_out = np.array(_lm, dtype=np.float64)
            else:
                lib_areas_out = np.atleast_1d(np.sum(Hidden_lib_areas)) if len(Hidden_lib_areas) else np.array([0.0])
                lib_one_area_out = np.atleast_1d(np.sum(Hid_one_area)) if len(Hid_one_area) else np.array([0.0])
                lib_one_areaConc_out = np.atleast_1d(np.sum(Hid_one_areaConc)) if len(Hid_one_areaConc) else np.array([0.0])
                lib_molarity_out = np.atleast_1d(np.sum(Hid_molarity)) if len(Hid_molarity) else np.array([0.0])
        else:
            lib_areas_out = np.atleast_1d(np.sum(Hidden_lib_areas)) if len(Hidden_lib_areas) else np.array([0.0])
            lib_one_area_out = np.atleast_1d(np.sum(Hid_one_area)) if len(Hid_one_area) else np.array([0.0])
            lib_one_areaConc_out = np.atleast_1d(np.sum(Hid_one_areaConc)) if len(Hid_one_areaConc) else np.array([0.0])
            lib_molarity_out = np.atleast_1d(np.sum(Hid_molarity)) if len(Hid_molarity) else np.array([0.0])

    elif len(rest_peaks_areas) > 0 and len(Hidden_LibPeakLocations) > 0:
        # ГБ одним пиком (гладкая/фаикс)
        if len(LibPeakLocations) > 0:
            lib_length = np.polyval(px, LibPeakLocations.astype(float))
            lib_peaksCorr = np.polyval(SDC, lib_length)
            maxLibValue = int(LibPeakLocations[0])
        else:
            lib_length = np.array([])
            lib_peaksCorr = np.array([])
        lib_areas_out = np.atleast_1d(np.sum(Hidden_lib_areas)) if len(Hidden_lib_areas) else np.array([0.0])
        lib_one_area_out = np.atleast_1d(np.sum(Hid_one_area)) if len(Hid_one_area) else np.array([0.0])
        lib_one_areaConc_out = np.atleast_1d(np.sum(Hid_one_areaConc)) if len(Hid_one_areaConc) else np.array([0.0])
        lib_molarity_out = np.atleast_1d(np.sum(Hid_molarity)) if len(Hid_molarity) else np.array([0.0])

    if len(lib_length) == 0 and len(LibPeakLocations) > 0:
        LibPeakLocations = np.array([], dtype=int)
        final_Lib_local_minimums_arr = np.array([], dtype=int)

    # --- Финальные массивы ---
    _s = lambda a: np.atleast_1d(np.asarray(a, dtype=float))
    all_areasConc = np.concatenate([[CONC_pair[0]], _s(lib_one_areaConc_out), [CONC_pair[-1]]])
    all_areas = np.concatenate([[st_areas_pair[0]], _s(lib_areas_out), [st_areas_pair[-1]]])
    molarity = np.concatenate([[st_molarity[0]], _s(lib_molarity_out), [st_molarity[-1]]])
    all_peaks = np.concatenate([[st_length[0]], LibPeakLocations, [st_length[-1]]])
    all_peaksCorr = np.concatenate([[st_peaksCorr[0]],
                                    _s(lib_peaksCorr) if len(lib_peaksCorr) else np.array([]),
                                    [st_peaksCorr[-1]]])

    t_final_locations = np.polyval(px, final_Lib_local_minimums_arr.astype(float)) \
        if len(final_Lib_local_minimums_arr) else np.array([])
    t_unrec = np.polyval(px, unrecognized_peaks.astype(float)) if len(unrecognized_peaks) else np.array([])
    unrec_corr = np.polyval(SDC, t_unrec) if len(t_unrec) else np.array([])

    maxLibPeak_out = None
    if maxLibValue is not None and np.isscalar(maxLibValue):
        idx_mlv = int(maxLibValue)
        if 0 <= idx_mlv < len(mainCorr):
            maxLibPeak_out = mainCorr[idx_mlv]

    totalLibArea = float(np.sum(lib_areas_out)) if len(lib_areas_out) else 0.0
    totalLibConc = float(np.sum(lib_one_areaConc_out)) if len(lib_one_areaConc_out) else 0.0
    totalLibMolarity = float(np.sum(lib_molarity_out)) if len(lib_molarity_out) else 0.0

    # --- Закраска ---
    x_fill_out = np.array([])
    y_fill_out = np.array([])
    x_Lib_fill_out = np.array([])
    y_Lib_fill_out = np.array([])

    if len(x_fill_1):
        ends = np.array([x_fill_1[0], x_fill_1[-1]], dtype=int)
        ends = np.clip(ends, 0, len(t_main) - 1)
        t_ends = t_main[ends]
        x_fill_out = np.linspace(t_ends[0], t_ends[1], 100)
        y_fill_out = np.interp(x_fill_out, t_main, denoised_data)

    if len(x_Lib_fill_1):
        ends2 = np.array([x_Lib_fill_1[0], x_Lib_fill_1[-1]], dtype=int)
        ends2 = np.clip(ends2, 0, len(t_main) - 1)
        t_ends2 = t_main[ends2]
        x_Lib_fill_out = np.linspace(t_ends2[0], t_ends2[1], 100)
        y_Lib_fill_out = np.interp(x_Lib_fill_out, t_main, denoised_data)

    hpx = np.round(lib_peaksCorr).astype(int) if len(lib_peaksCorr) else np.array([], dtype=int)
    unr = np.round(unrec_corr).astype(int) if len(unrec_corr) else np.array([], dtype=int)
    stp = np.array([LIZ[0], LIZ[-1]])

    return {
        't_main': t_main,
        'denoised_data': denoised_data,
        'st_peaks': st_peaks,
        'st_length': st_length,
        't_unrecognized_peaks': t_unrec,
        'unrecognized_peaks': unrecognized_peaks,
        'lib_length': lib_length,
        'LibPeakLocations': LibPeakLocations,
        't_final_locations': t_final_locations,
        'final_Lib_local_minimums': final_Lib_local_minimums_arr,
        'hpx': hpx,
        'unr': unr,
        'stp': stp,
        'all_areas': all_areas,
        'all_peaksCorr': all_peaksCorr,
        'all_peaks': all_peaks,
        'all_areasConc': all_areasConc,
        'molarity': molarity,
        'maxLibPeak': maxLibPeak_out,
        'maxLibValue': maxLibValue,
        'totalLibArea': totalLibArea,
        'totalLibConc': totalLibConc,
        'totalLibMolarity': totalLibMolarity,
        'x_fill': x_fill_out,
        'y_fill': y_fill_out,
        'x_Lib_fill': x_Lib_fill_out,
        'y_Lib_fill': y_Lib_fill_out,
    }


   
    

    




# ============================================================================
#  Внутренние вспомогательные функции
# ============================================================================

def _select_repers_by_points(Points, peak_locs, peak, orders, max_order_val):
    """
    Выбор первого/второго реперов по баллам Points (ветка 3 или повторная
    логика из ветки 2).
    """
    max_pt = int(np.floor(np.max(Points)))
    idx_main = np.where(np.floor(Points) == max_pt)[0]
    peak_for_choose = idx_main.copy()
    current_val = max_pt - 1

    while len(peak_for_choose) < 5 and current_val >= 0:
        extra = np.where(np.floor(Points) == current_val)[0]
        peak_for_choose = np.unique(np.concatenate([peak_for_choose, extra]))
        current_val -= 1

    k = 100
    first_reper_idx = np.where(np.abs(peak_locs.astype(float) - peak[0]) <= k)[0]
    pre_first = peak_for_choose[np.isin(peak_for_choose, first_reper_idx)]
    pre_second = np.setdiff1d(peak_for_choose, pre_first)

    while len(pre_second) == 0 and current_val >= 0:
        extra = np.where(np.floor(Points) == current_val)[0]
        peak_for_choose = np.unique(np.concatenate([peak_for_choose, extra]))
        current_val -= 1

        first_reper_idx = np.where(np.abs(peak_locs.astype(float) - peak[0]) <= k)[0]
        pre_first = peak_for_choose[np.isin(peak_for_choose, first_reper_idx)]
        pre_second = np.setdiff1d(peak_for_choose, pre_first)

        if len(peak_for_choose) > 10:
            while len(peak_for_choose) > 10 and k > 50:
                k -= 10
                first_reper_idx = np.where(
                    np.abs(peak_locs.astype(float) - peak[0]) <= k)[0]
                pre_first = peak_for_choose[np.isin(peak_for_choose, first_reper_idx)]
                pre_second = np.setdiff1d(peak_for_choose, pre_first)
                if len(pre_second):
                    break

            min_pt = int(np.min(np.floor(Points[peak_for_choose])))
            idx_rm = np.where(np.floor(Points) == min_pt)[0]
            peak_for_choose = np.setdiff1d(peak_for_choose, idx_rm)

            first_reper_idx = np.where(
                np.abs(peak_locs.astype(float) - peak[0]) <= k)[0]
            pre_first = peak_for_choose[np.isin(peak_for_choose, first_reper_idx)]
            pre_second = np.setdiff1d(peak_for_choose, pre_first)

    return pre_first, pre_second

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
