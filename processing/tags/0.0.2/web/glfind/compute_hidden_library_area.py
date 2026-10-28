import numpy as np
from numpy.typing import NDArray

from lib.matlab.round import matlab_round


def compute_hidden_library_area(
    corrected_signal: NDArray[np.floating],
    start_index: np.integer,
    end_index: np.integer,
    max_peak_idx: np.integer,
) -> tuple[
    NDArray[np.integer],
    NDArray[np.floating],
    NDArray[np.integer],
]:
    """Вычисление расположения и площади скрытого пика библиотеки"""

    left_median = (start_index + max_peak_idx) / 2  # Находим медианное значение между start_index и maxLibValue (левая середина пика ГБ)
    right_median = (max_peak_idx + end_index) / 2  # Находим медианное значение между maxLibValue и end_index (правая середина пика ГБ)

    left_idx = matlab_round(left_median)
    right_idx = matlab_round(right_median)

    # Создаем массив чисел от первого округленного значения до второго с шагом 1
    library_peak_range = np.arange(left_idx, right_idx + 1)

    # TODO: Почему добавляем индекс начала?
    hidden_final_lib_local_minimums = np.concatenate(([start_index], library_peak_range))  # все площади

    hidden_lib_areas: list[float] = []
    # ДЛЯ ЗАКРАСКИ И ОБЩЕЙ ПЛОЩАДИ
    for i in range(len(hidden_final_lib_local_minimums) - 1):
        start_idx = hidden_final_lib_local_minimums[i]
        end_idx = hidden_final_lib_local_minimums[i + 1]
        # TODO: Поиск площади методом Симпсона
        area = float(np.trapezoid(corrected_signal[start_idx:end_idx + 1]))
        hidden_lib_areas.append(area)

    return library_peak_range, np.array(hidden_lib_areas, dtype=np.float64), hidden_final_lib_local_minimums
