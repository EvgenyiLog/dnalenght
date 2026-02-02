import numpy as np
from numpy.typing import NDArray


def compute_smooth_library_concentrations(
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
    """Вычисление суммарных характеристик (площадь, концентрацию, молярность) для гладкой геномной библиотеки"""
    lib_areas = np.array([np.sum(hidden_lib_areas)], dtype=np.float64)
    lib_one_area = np.array([np.sum(hid_one_area)], dtype=np.float64)
    lib_one_area_conc = np.array([np.sum(hid_one_area_conc)], dtype=np.float64)
    lib_molarity = np.array([np.sum(hid_molarity)], dtype=np.float64)

    return (
        lib_areas,
        lib_one_area,
        lib_one_area_conc,
        lib_molarity,
    )
