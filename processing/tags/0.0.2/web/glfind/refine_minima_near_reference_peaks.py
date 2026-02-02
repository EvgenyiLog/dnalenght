import numpy as np
from numpy.typing import NDArray


def refine_minima_near_reference_peaks(
    minima_candidates: NDArray[np.integer],
    reference_peaks: NDArray[np.integer],
    unrecognized_peaks: NDArray[np.integer],
    corrected_signal: NDArray[np.floating],
) -> NDArray[np.integer]:
    """Уточнение списка минимумов, проверяя расстояние от реперных пиков до ближайших минимумов"""

    additional_minima = []

    # Проверка минимумов (массив complete_Peaks_Locations): если ближайший минимум дальше, чем 10, то переназначаем все минимумы текущего пика на 7
    for ref_idx in reference_peaks:
        # Ищем ближайшее значение в массиве complete_Peaks_Locations
        left = np.max(minima_candidates[minima_candidates < ref_idx])
        right = np.min(minima_candidates[minima_candidates > ref_idx])
        closest_value = [left, right]

        idx = np.abs(closest_value - ref_idx)

        # Проверяем расстояние с двух сторон
        if np.any(idx > 10):
            # Если хотя бы одно расстояние превышает 10, добавляем значения в min_st_length
            additional_minima.append(ref_idx - 7)
            additional_minima.append(ref_idx + 7)

    # Объединяем массивы, удаляем дубликаты и сортируем массив
    refined_minima = np.union1d(minima_candidates, np.array(additional_minima, dtype=np.int64))

    # На случай, если репер и неопознанный пик находятся слишком близко (расстояние < 10) - иначе впоследствии программа не понимает,
    # какой минимум между ними должен быть (их минимумы накладываются)
    for ref_idx in reference_peaks:
        # Поиск значений в pre_unrecognized_peaks, лежащих в пределах [current_st_length - 10, current_st_length + 10]
        close_unrecognized = unrecognized_peaks[(unrecognized_peaks >= (ref_idx - 10)) & (unrecognized_peaks <= (ref_idx + 10))]

        for unrec_idx in close_unrecognized:
            # Определение диапазона в denoised_data для поиска минимума
            start = min(ref_idx, unrec_idx)
            end = max(ref_idx, unrec_idx)

            # Поиск наименьшего значения в указанном диапазоне и его индекса
            min_idx = start + np.argmin(corrected_signal[start:end + 1])

            # Удаление значений из complete_Peaks_Locations, лежащих в этом диапазоне
            refined_minima = refined_minima[
                (refined_minima < start) | (refined_minima > end)
            ]
            # Добавление найденного индекса в complete_Peaks_Locations
            refined_minima = np.union1d(refined_minima, min_idx)

    return refined_minima
