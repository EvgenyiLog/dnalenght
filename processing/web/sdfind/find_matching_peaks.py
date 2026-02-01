import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


def find_matching_peaks(
    denoised_signal: NDArray[np.floating],
    standard_sizes: NDArray[np.floating],
    release_times: NDArray[np.floating],
) -> NDArray[np.integer]:
    """Находит пики, соответствующие калибровочным стандартам."""

    denoised_flipped = np.flip(denoised_signal)

    # Понадобится для отсеивания найденных пиков в соотвествии с выбранным законом
    poly_coef = np.polyfit(standard_sizes, release_times, 4)  # полином 4 степени
    new_sizes = np.polyval(poly_coef, np.flip(standard_sizes))
    size_deltas = np.abs(np.diff(new_sizes))

    overmuch = len(standard_sizes) * 2.4  # порог количества, значение взято из опыта
    threshold = np.quantile(denoised_flipped, 0.995, method='hazen')  # для начала возьмем порог на уровне 99.5%, будем его снижать, если надо

    # *** НАЙДЕМ В СПЕКТРЕ ПИКИ, СООТВЕТВУЮЩИЕ ПИКАМ СТАНДАРТА ***
    for _ in range(30):   # главный цикл (30 попыток)
        # ищем пики, пытаемся среди найденных отобрать подходящие, если не
        # удалось - снижаем порог и повторяем процедуру

        # ** ИЩЕМ ПИКИ В СПЕКТРЕ **
        threshold *= 0.9

        peaks = np.empty(0, dtype=np.int64)
        for _ in range(20):
            peaks = find_peaks(denoised_flipped, height=threshold, distance=9)[0]  # Equal MinPeakDistance=8
            if len(peaks) >= len(standard_sizes):
                break
            threshold *= 0.9

        if len(peaks) >= overmuch:
            break

        #  ОТСЕИВАЕМ ЛИШНИЕ
        for k in range(len(standard_sizes) - 1):
            for j in range(k + 1, len(peaks)):
                base_step = (peaks[j] - peaks[k]) / size_deltas[0]  # кандидат на "базовый шаг"
                step = base_step  # pace - текущий шаг
                filtered_peaks = [0]  # кандидат

                liz_idx, peak_idx, next_pos = 0, 0, 0
                # проверим является ли кандидат на "базовый шаг" настоящим
                while next_pos < peaks[-1] and liz_idx < len(standard_sizes) - 1:
                    prev_pos = peaks[peak_idx]
                    delta = step * size_deltas[liz_idx]
                    next_pos = prev_pos + delta
                    dists = np.abs(peaks - next_pos)
                    nearest_idx = np.argmin(dists)   # индекс ближайшего значения
                    dist = dists[nearest_idx]

                    if dist < delta / 2:  # пик лежит примерно там где и ожидалось
                        peak_idx = int(nearest_idx)
                        filtered_peaks.append(peak_idx)
                        step = (peaks[nearest_idx] - prev_pos) / size_deltas[liz_idx]  # фактический шаг
                        liz_idx += 1
                    else:  # пика в ожидаемом месте нет - возможно начальный пик ложный
                        peak_idx = filtered_peaks[0] + 1  # примем следующий пик за начальный
                        filtered_peaks = [peak_idx]
                        step = base_step
                        liz_idx = 0  # стандарт начнем с начала

                if len(filtered_peaks) == len(standard_sizes):  # нужное количество пиков нашли - выходим из цикла
                    matching_peaks_flipped = peaks[filtered_peaks]
                    matching_peaks = np.flip(-matching_peaks_flipped + len(denoised_signal) - 1)
                    return matching_peaks

    return np.empty(0, dtype=np.int64)
