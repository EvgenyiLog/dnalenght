import numpy as np
from numpy.typing import NDArray


def compute_fragmented_library_concentrations(
    lib_peak_locations: NDArray[np.integer],
    hidden_lib_peak_locations: NDArray[np.integer],
    hidden_lib_areas: NDArray[np.floating],
    hid_one_area: NDArray[np.floating],
    hid_one_area_conc: NDArray[np.floating],
    hid_molarity: NDArray[np.floating],
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """Вычисление суммарных характеристик (площадь, концентрацию, молярность) для библиотеки с выраженными локальными пиками"""

    # Находим индексы элементов LibPeakLocations в Hidden_LibPeakLocations
    # TODO searchsorted не фильтрует отсутствующие элементы! Нужно заменить на правильную реализацию
    indices = np.searchsorted(hidden_lib_peak_locations, lib_peak_locations)

    # Суммируем площади по индексам
    lib_areas = np.empty(0, dtype=np.float64)
    lib_one_area = np.empty(0, dtype=np.float64)
    lib_one_area_conc = np.empty(0, dtype=np.float64)
    lib_molarity = np.empty(0, dtype=np.float64)
    prev_idx = 0
    # Так как Hidden_LibPeakLocations - это куски фрагментов по 1 пн, а LibPeakLocations - это фрагменты, которые были
    # найдены (нужно для визуализации), нам нужно найти концентрацию этих фрагментов: концентрация каждого
    # фрагмента - сумма предыдущего площадей Hidden_LibPeakLocations
    for current_idx in indices:
        lib_areas = np.append(lib_areas, np.sum(hidden_lib_areas[prev_idx:current_idx + 1]))  # площадь
        lib_one_area = np.append(lib_one_area, np.sum(hid_one_area[prev_idx:current_idx + 1]))  # площадь на 1 пн
        lib_one_area_conc = np.append(lib_one_area_conc, np.sum(hid_one_area_conc[prev_idx:current_idx + 1]))  # концентрация
        lib_molarity = np.append(lib_molarity, np.sum(hid_molarity[prev_idx:current_idx + 1]))  # молярность
        prev_idx = current_idx + 1  # Начинаем следующий интервал с нового индекса +1

    return (
        lib_areas,
        lib_one_area,
        lib_one_area_conc,
        lib_molarity,
    )
