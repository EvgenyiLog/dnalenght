from __future__ import annotations

import warnings
from typing import Dict, Any, Optional,Tuple

import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_prominences, peak_widths
import traceback

import numpy as np
from scipy.signal import savgol_filter


import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import gaussian_filter1d

def _score_and_select(data, peaks, n_target, dLIZ):
    """
    Балльная оценка пиков и выбор n_target лучших.
    """
    n_peaks = len(peaks)
    if n_peaks < n_target:
        return None

    # Расчёт характеристик
    try:
        prominences, _, _ = peak_prominences(data, peaks)
        widths, _, _, _ = peak_widths(data, peaks, rel_height=0.5)
    except Exception:
        return None

    heights = data[peaks]
    Points = np.zeros(n_peaks, dtype=np.float64)

    # === ЭТАП 1: Площадь / высота пика ===
    if n_peaks > 1:
        h_range = heights.max() - heights.min()
        if h_range > 0:
            norm_h = (heights - heights.min()) / h_range
        else:
            norm_h = np.ones(n_peaks)
        Points += norm_h
    else:
        Points += 1.0

    # === ЭТАП 2: Проминенс ===
    if n_peaks > 1:
        sorted_prom_idx = np.argsort(prominences)[::-1]
        p_max_idx = sorted_prom_idx[0]
        Points[p_max_idx] += 1.0

        remaining = sorted_prom_idx[1:]
        if len(remaining) > 0:
            rem_prom = prominences[remaining]
            p_range = rem_prom.max() - rem_prom.min()
            if p_range > 0:
                norm_p = ((rem_prom - rem_prom.min()) / p_range) * 0.95
            else:
                norm_p = np.full(len(remaining), 0.5)
            for i, idx in enumerate(remaining):
                Points[idx] += norm_p[i]

    # === ЭТАП 3: Соотношение высота/ширина ===
    safe_w = np.where(widths == 0, 1e-10, widths)
    R = heights / safe_w
    if n_peaks > 1:
        r_range = R.max() - R.min()
        if r_range > 0:
            norm_r = (R - R.min()) / r_range
        else:
            norm_r = np.ones(n_peaks) * 0.5
        Points += norm_r

    # === ЭТАП 4: Штраф за слишком широкие пики ===
    if n_peaks > 1:
        median_w = np.median(widths)
        for i in range(n_peaks):
            if widths[i] > 3 * median_w:
                Points[i] -= 0.5

    # === Выбор лучших n_target пиков ===
    sorted_indices = np.argsort(Points)[::-1]
    selected = np.sort(sorted_indices[:n_target])

    # === Проверка порядка и расстояний ===
    selected_pos = peaks[selected]

    # Попробуем улучшить: если расстояния не соответствуют dLIZ,
    # попробуем другие комбинации
    best_score = _validate_spacing(selected_pos, dLIZ)
    best_selected = selected_pos.copy()

    if best_score < 0.5 and n_peaks > n_target:
        # Перебор с учётом расстояний dLIZ
        spacing_result = _select_by_spacing(peaks, n_target, dLIZ, Points)
        if spacing_result is not None:
            sp_score = _validate_spacing(spacing_result, dLIZ)
            if sp_score > best_score:
                best_selected = spacing_result
                best_score = sp_score

    if best_score < 0.2:
        return None

    return best_selected.astype(int)


def _snap_peaks_to_local_max(data, peaks, radius=4):
    """
    Для каждого пика просматривает окрестность ±radius точек
    и сдвигает пик к позиции локального максимума.
    """
    n = len(data)
    corrected = np.copy(peaks)
    for i, pk in enumerate(peaks):
        lo = max(0, int(pk) - radius)
        hi = min(n, int(pk) + radius + 1)
        corrected[i] = lo + int(np.argmax(data[lo:hi]))
    return corrected


def _validate_spacing(positions, dLIZ):
    """
    Оценивает, насколько расстояния между выбранными пиками
    соответствуют ожидаемым dLIZ. Возвращает score 0..1.
    """
    if len(positions) < 2 or len(dLIZ) < 1:
        return 0.0

    actual_diff = np.diff(positions.astype(float))
    if len(actual_diff) == 0:
        return 0.0

    # Нормализуем: вычисляем отношение actual/expected
    n_common = min(len(actual_diff), len(dLIZ))
    if n_common == 0:
        return 0.0

    expected = dLIZ[:n_common]
    actual = actual_diff[:n_common]

    # Рассчитываем масштабный коэффициент
    safe_exp = np.where(expected == 0, 1e-10, expected)
    ratios = actual / safe_exp

    # Оценка: все отношения должны быть примерно одинаковы
    if len(ratios) < 2:
        return 0.5

    mean_ratio = np.mean(ratios)
    if mean_ratio <= 0:
        return 0.0

    # Коэффициент вариации — чем меньше, тем лучше
    cv = np.std(ratios) / mean_ratio if mean_ratio > 0 else 1.0
    score = max(0.0, 1.0 - cv)
    return score


def _select_by_spacing(peaks, n_target, dLIZ, Points):
    """
    Жадный алгоритм: выбирает пики с учётом ожидаемых расстояний dLIZ.
    """
    if len(peaks) < n_target or len(dLIZ) < n_target - 1:
        return None

    peaks_f = peaks.astype(float)
    best_result = None
    best_score = -1

    # Пробуем каждый пик как стартовый (из top-N по баллам)
    top_n = min(15, len(peaks))
    top_indices = np.argsort(Points)[::-1][:top_n]

    for start_rank in range(min(top_n, len(peaks))):
        start_idx = top_indices[start_rank]
        selected = [start_idx]

        # Оцениваем базовый шаг из среднего расстояния
        # Пробуем несколько масштабов
        for scale_candidate_idx in range(start_rank + 1, min(top_n, len(peaks))):
            candidate_idx = top_indices[scale_candidate_idx]
            if candidate_idx == start_idx:
                continue

            pos1 = peaks_f[start_idx]
            pos2 = peaks_f[candidate_idx]
            if pos2 <= pos1:
                continue

            base_step = (pos2 - pos1) / dLIZ[0] if dLIZ[0] != 0 else 0
            if base_step <= 0:
                continue

            # Жадный поиск
            selected = [start_idx]
            current_pos = pos1
            for d_idx in range(len(dLIZ)):
                if d_idx >= n_target - 1:
                    break
                expected_pos = current_pos + base_step * dLIZ[d_idx]
                distances = np.abs(peaks_f - expected_pos)
                closest = int(np.argmin(distances))
                tolerance = base_step * dLIZ[d_idx] * 0.4
                if distances[closest] < tolerance and closest not in selected:
                    selected.append(closest)
                    current_pos = peaks_f[closest]
                    # Адаптируем шаг
                    if dLIZ[d_idx] != 0:
                        base_step = (peaks_f[closest] - current_pos + base_step * dLIZ[d_idx]) / (2 * dLIZ[d_idx])
                else:
                    break

            if len(selected) == n_target:
                result = peaks[np.sort(selected)]
                score = _validate_spacing(result, dLIZ)
                if score > best_score:
                    best_score = score
                    best_result = result

    return best_result


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


def autofind_peaks(data, LIZ, dLIZ):
    """
    Авто-поиск пиков стандарта длин в сигнале.

    Параметры
    ---------
    data : 1-D array — обработанный (развёрнутый) сигнал стандарта длин
    LIZ  : 1-D array — длины фрагментов стандарта (развёрнутые)
    dLIZ : 1-D array — ожидаемые расстояния между пиками

    Возвращает
    ----------
    locs : 1-D int array или None — позиции найденных пиков (len == len(LIZ)),
           или None если не удалось найти
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    n_liz = len(LIZ)

    if n_liz < 2 or len(data) < 20:
        return None

    # 1. Предварительная обработка
    data_smooth = gaussian_filter1d(data.copy(), sigma=1.5)
    data_smooth[data_smooth < 0] = 0

    # 2. Адаптивный поиск пиков с несколькими уровнями порога
    best_result = None
    best_score = -1

    for height_frac in [0.05, 0.03, 0.02, 0.01, 0.005]:
        for dist_factor in [5, 7, 10, 3]:
            height_threshold = height_frac * np.max(data_smooth)
            distance = max(3, dist_factor)

            try:
                peaks, _ = find_peaks(data_smooth,
                                      height=height_threshold,
                                      distance=distance,
                                      prominence=0.005 * np.max(data_smooth))
            except Exception:
                continue

            if len(peaks) < n_liz:
                continue

            # 3. Балльная система оценки
            result = _score_and_select(data_smooth, peaks, n_liz, dLIZ)
            if result is not None:
                score = _validate_spacing(result, dLIZ)
                if score > best_score:
                    best_score = score
                    best_result = result

    # Коррекция: смещение каждого пика к локальному максимуму ±4 точки
    # на оригинальных (не сглаженных) данных
    if best_result is not None:
        best_result = _snap_peaks_to_local_max(data, best_result, radius=4)

    return best_result


def _empty_result(raw_ref):
    """Пустой результат."""
    ref_out = raw_ref[::-1] if len(raw_ref) else raw_ref
    return (np.array([], dtype=int),
            np.array([], dtype=np.float64),
            ref_out,
            np.array([], dtype=np.float64))


def _recalc_areas(locs2, crossings, raw_ref, n_liz):
    """
    Пересчитать площади: для каждого пика найти ближайшие
    границы из crossings и вычислить площадь.
    """
    n_raw = len(raw_ref)
    area_list = []
    for pk in locs2:
        pk = int(pk)
        left_bounds = crossings[crossings < pk]
        right_bounds = crossings[crossings > pk]
        if len(left_bounds) == 0 or len(right_bounds) == 0:
            area_list.append(0.0)
            continue
        x_start = int(left_bounds[-1])
        x_end = int(right_bounds[0])
        if 0 <= x_start < x_end < n_raw:
            area_list.append(float(np.trapezoid(raw_ref[x_start:x_end + 1])))
        else:
            area_list.append(0.0)
    return np.array(area_list[::-1], dtype=np.float64)

# ======================================================================
# ============================ MAIN ====================================
# ======================================================================

def sdfind(
    raw_ref: np.ndarray,
    in_liz: np.ndarray,
    conc: np.ndarray,
    locs_pol: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Поиск пиков стандарта длин и расчёт площадей/молярности.

    Параметры
    ---------
    raw_ref  : 1-D array — сырые данные стандарта длин
    in_liz   : 1-D array — длины фрагментов стандарта (в пн)
    locs_pol : 1-D array — времена выхода (ReleaseTime) для полин. подгонки
    conc     : 1-D array — концентрации фрагментов стандарта

    Возвращает
    ----------
    locs2       : 1-D int array — позиции найденных пиков (0-based, слева направо)
    area        : 1-D float array — площади под пиками
    raw_ref     : 1-D float array — обработанные данные (развёрнуты обратно)
    sd_molarity : 1-D float array — молярности (нмоль/л)
    """
   
    
    raw_ref = np.asarray(raw_ref, dtype=np.float64).ravel()
    in_liz = np.asarray(in_liz, dtype=np.float64).ravel()
    locs_pol = np.asarray(locs_pol, dtype=np.float64).ravel()
    conc = np.asarray(conc, dtype=np.float64).ravel()

    LIZ = in_liz[::-1]  # развёрнутый вектор фрагментов
    n_liz = len(LIZ)

    if n_liz < 2:
        return _empty_result(raw_ref)

    x = np.arange(len(raw_ref), dtype=np.float64)

    
    zr_ref = raw_ref

    # --- Разворот массивов (как MATLAB flipud) ---
    zr_ref = zr_ref[::-1].copy()
    raw_ref = raw_ref[::-1].copy()

    # --- Полиномиальная аппроксимация ожидаемых позиций ---
    # Если locs_pol все нулевые — использовать линейную оценку
    if np.all(locs_pol == 0) or len(locs_pol) == 0:
        n_pts = len(zr_ref)
        locs_pol = np.linspace(n_pts * 0.1, n_pts * 0.9, n_liz)

    # === ПРОВЕРКА: Длины должны совпадать ===
    if len(in_liz) != len(locs_pol):
        min_len = min(len(in_liz), len(locs_pol))
        
        # Обрезаем до одинаковой длины (берем первые min_len элементов)
        print(f"WARNING: Обрезаем массивы до длины {min_len}")
        in_liz = in_liz[:min_len]
        locs_pol = locs_pol[:min_len]
        z = np.polyfit(in_liz, locs_pol, 1)
    else:
        z = np.polyfit(in_liz, locs_pol, min(4, n_liz - 1))
    new_LIZ = np.polyval(z, LIZ)
    dLIZ = np.abs(np.diff(new_LIZ))  # length = n_liz - 1

    if len(dLIZ) == 0 or dLIZ[0] == 0:
        return _empty_result(raw_ref)

    threshold = np.quantile(zr_ref, 0.995)

    data_peak_idx = []
    found = False
    locs2 = np.array([], dtype=int)

    # === Главный цикл: 30 попыток со снижением порога ===
    for _thresh_loop in range(30):
        threshold *= 0.9

        # Понижаем порог, пока не найдём достаточно пиков
        for _tc in range(20):
            pks, locs, _, _ = findpeaks_matlab(zr_ref,
                                               min_peak_height=threshold,
                                               min_peak_distance=7)
            if len(locs) < n_liz:
                threshold *= 0.9
            else:
                break

        overmuch = 2.4
        if len(pks) >= overmuch * n_liz:
            # Слишком много пиков — продолжаем снижать порог
            continue

        if len(locs) < 2:
            continue

        # Предвычисляем float-версию locs
        locs_f = locs.astype(np.float64)
        locs_last = locs_f[-1]
        dLIZ_0 = dLIZ[0]

        # --- Отсеивание лишних пиков «шаговым» алгоритмом ---
        # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: dataPeakIdx ВСЕГДА начинается с [0]
        # (MATLAB: dataPeakIdx=[1]; iDat=1;), а не с [k].
        # Внешний цикл k только задаёт пару для вычисления PACE.
        for k in range(min(n_liz - 1, len(locs))):
            found_inner = False
            for j in range(k + 1, len(locs)):
                if dLIZ_0 == 0:
                    continue
                PACE = (locs_f[j] - locs_f[k]) / dLIZ_0
                if PACE <= 0:
                    continue
                pace = PACE

                # Начинаем ВСЕГДА с первого пика (индекс 0)
                data_peak_idx = [0]
                i_liz = 0
                i_dat = 0
                D_next = 0.0

                while D_next < locs_last and i_liz < n_liz - 1:
                    D_prev = locs_f[i_dat]
                    if i_liz >= len(dLIZ):
                        break
                    Dd = pace * dLIZ[i_liz]
                    if Dd <= 0:
                        break
                    D_next = D_prev + Dd

                    # Поиск ближайшего пика
                    dst = np.abs(locs_f - D_next)
                    idx_min = int(np.argmin(dst))
                    min_dist = dst[idx_min]

                    if min_dist < Dd * 0.5:
                        data_peak_idx.append(idx_min)
                        if dLIZ[i_liz] != 0:
                            pace = (locs_f[idx_min] - D_prev) / dLIZ[i_liz]
                        i_liz += 1
                        i_dat = idx_min
                    else:
                        # Пика нет — сдвигаем стартовую точку
                        new_start = data_peak_idx[0] + 1
                        if new_start >= len(locs):
                            break
                        i_dat = new_start
                        data_peak_idx = [new_start]
                        pace = PACE
                        i_liz = 0

                if len(data_peak_idx) == n_liz:
                    found_inner = True
                    break
            if found_inner and len(data_peak_idx) == n_liz:
                break

        if len(data_peak_idx) == n_liz:
            found = True
            break

    # === Fallback: авто-поиск через AutoFind ===
    if not (found and len(data_peak_idx) == n_liz):
        try:
            
            auto_locs = autofind_peaks(zr_ref, LIZ, dLIZ)
            if auto_locs is not None and len(auto_locs) == n_liz:
                locs2 = auto_locs.astype(int)
                found = True
                print(f'SDFind1_2: основной алгоритм не нашёл стандарт, '
                      f'авто-поиск нашёл {n_liz} пиков.', flush=True)
        except Exception as e:
            print(f'SDFind1_2: авто-поиск недоступен: {e}', flush=True)

    if not found:
        return _empty_result(raw_ref)

    # Если нашли основным алгоритмом
    if len(locs2) == 0 and len(data_peak_idx) == n_liz:
        locs2 = locs[data_peak_idx]

    # --- Нахождение минимумов ---
    filt_zr = savgol_filter(zr_ref, 3, 1)
    crossings2 = np.where(np.diff((filt_zr > 0).astype(np.int8)))[0]
    flip_data = -filt_zr
    _, peak_locs1, _, _ = findpeaks_matlab(flip_data)
    crossings = np.union1d(peak_locs1, crossings2)

    if len(crossings) == 0:
        return _empty_result(raw_ref)

    # === Расчёт площадей ===
    area_list = []
    n_raw = len(raw_ref)

    for i in range(len(crossings) - 1):
        c_lo, c_hi = int(crossings[i]), int(crossings[i + 1])
        between = (locs2 >= c_lo) & (locs2 <= c_hi)
        if int(np.sum(between)) == 1:
            x_start = c_lo
            x_end = c_hi
            if 0 <= x_start < x_end < n_raw:
                area_list.append(float(np.trapezoid(raw_ref[x_start:x_end + 1])))

    area = np.array(area_list[::-1], dtype=np.float64)

    if len(area) != n_liz:
        # Запасной метод: пересчёт площадей для каждого пика
        area = _recalc_areas(locs2, crossings, raw_ref, n_liz)
        if len(area) != n_liz:
            print(f'Стандарт длин: площади не совпадают ({len(area)} ≠ {n_liz}).', flush=True)
            return _empty_result(raw_ref)

    # --- Развернуть обратно (как MATLAB) ---
    # MATLAB: locs2 = -locs2 + length(rawRef) + 1; locs2 = flipud(locs2);
    n_pts = len(raw_ref)
    raw_ref = raw_ref[::-1]
    locs2 = (n_pts - 1 - locs2)[::-1]

    # --- Молярность ---
    # MATLAB: SD_molarity = ((CONC * 10^(-3)) ./ (649 * flipud(LIZ))) * 10^9
    LIZ_orig = in_liz
    sd_molarity = ((conc * 1e-3) / (649.0 * LIZ_orig)) * 1e9

    return locs2.astype(int), area.astype(float), raw_ref, sd_molarity.astype(float)    
