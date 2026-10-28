import numpy as np
from numpy.typing import NDArray


def select_reference_peaks(
    peaks: NDArray[np.integer],
    expected_spacing: np.integer,
    corrected_signal: NDArray[np.floating],
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    """Нахождение двух реперных пиков на основе ожидаемого расстояния между ними"""

    if len(peaks) < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    start_peak_idx = peaks[0]
    recognized_peaks = []
    rejected_peaks = []

    # Поиск второго пика
    for cur_peak_idx in peaks[1:]:
        # Вычитание базового значения
        distance = cur_peak_idx - start_peak_idx
        # Проверка расстоянияs
        if np.abs(distance - expected_spacing) <= 0.2 * expected_spacing:  # Используем 20% допуск
            recognized_peaks.append(cur_peak_idx)
        else:
            rejected_peaks.append(cur_peak_idx)

    if len(recognized_peaks) == 0:
        end_peak = rejected_peaks.pop()  # Последний нераспознанный, удаляем из нераспознанных
    elif len(recognized_peaks) == 1:
        end_peak = recognized_peaks[0]
    else:
        # Создаем массив для хранения площадей
        areas = []

        # Перебираем все значения, начиная со второго
        for peak_idx in recognized_peaks:
            # Определяем границы (±7)
            left_idx = peak_idx - 7
            right_idx = peak_idx + 7

            # Интегрируем площадь между границами
            area = np.trapezoid(corrected_signal[left_idx:right_idx + 1])
            areas.append(area)

        # Находим индекс наибольшей площади
        max_area_idx = np.argmax(areas)
        # Выбираем как второй реперный пик
        end_peak = recognized_peaks[max_area_idx]

    return np.array([start_peak_idx, end_peak], dtype=np.int64), np.array(rejected_peaks, dtype=np.int64)
