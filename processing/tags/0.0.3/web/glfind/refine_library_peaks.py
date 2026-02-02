import numpy as np
from numpy.typing import NDArray


def refine_library_peaks(
    library_peaks: NDArray[np.integer],
    hidden_lib_peak_locations: NDArray[np.integer],
    final_lib_local_minimums: NDArray[np.integer],
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    # Проверка на то, лежат ли LibPeakLocations в пределах Hidden_LibPeakLocations: если нет, то удаляются
    in_bounds_mask = (library_peaks >= hidden_lib_peak_locations[0]) & (library_peaks <= hidden_lib_peak_locations[-1])
    # Удаляем все элементы check_LibPeakLocations из LibPeakLocations
    refined_library_peaks = library_peaks[in_bounds_mask]

    refined_lib_local_minima = np.copy(final_lib_local_minimums)

    # Добавляем самое первое значение Hidden_LibPeakLocations в качестве первой границы
    left_boundary = hidden_lib_peak_locations[0]
    if np.any(library_peaks < left_boundary):
        refined_library_peaks = np.insert(refined_library_peaks, 0, left_boundary)
        left_candidates = max(refined_lib_local_minima[refined_lib_local_minima < left_boundary])
        refined_lib_local_minima = refined_lib_local_minima[refined_lib_local_minima >= left_candidates]

    # Аналогично для второй границы
    right_boundary = hidden_lib_peak_locations[-1]
    if np.any(library_peaks > right_boundary):
        refined_library_peaks = np.append(refined_library_peaks, right_boundary)
        right_candidates = min(refined_lib_local_minima[refined_lib_local_minima > right_boundary])
        refined_lib_local_minima = refined_lib_local_minima[refined_lib_local_minima <= right_candidates]

    # Чистим повторы и сортируем массив
    refined_library_peaks = np.unique(refined_library_peaks)

    return refined_library_peaks, refined_lib_local_minima
