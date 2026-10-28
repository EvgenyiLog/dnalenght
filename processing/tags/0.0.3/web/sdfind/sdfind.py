from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from lib.matlab.wden import wden
from lib.matlab.msbackadj import msbackadj
from lib.sdfind.compute_peak_areas import compute_peak_areas
from lib.sdfind.find_matching_peaks import find_matching_peaks


@dataclass
class SDFindResult:
    corrected_signal: NDArray[np.floating]
    peaks: NDArray[np.integer]
    peak_areas: NDArray[np.floating]
    concentrations: NDArray[np.floating]
    molarity: NDArray[np.floating]
    size_fit: NDArray[np.floating]
    peak_fit: NDArray[np.floating]


def sdfind(
    raw_signal: NDArray[np.floating],
    standard_sizes: NDArray[np.floating],
    release_times: NDArray[np.floating],
    standard_conc: NDArray[np.floating],
) -> SDFindResult:
    x = np.arange(len(raw_signal))
    corrected_signal = msbackadj(x, raw_signal, window_size=140, step_size=40, quantile_value=0.1)  # коррекция бейзлайна
    denoised_signal = wden(corrected_signal, 'sqtwolog', 's', 'sln', 1, 'sym2')  # фильтр данных

    matching_peaks = find_matching_peaks(denoised_signal, standard_sizes, release_times)

    if len(matching_peaks) != len(standard_sizes):
        raise ValueError('Не удалось найти подходящее количество пиков. Проверьте калибровку стандартов длины.')

    peak_areas = compute_peak_areas(corrected_signal, denoised_signal, matching_peaks)

    if len(peak_areas) != len(standard_sizes):
        raise ValueError("Количество рассчитанных площадей не совпадает с количеством калибровочных стандартов.")

    concentrations = peak_areas / (standard_sizes ** 2 / 100)
    molarity = ((standard_conc * 1e-3) / (649 * standard_sizes)) * 1e9

    # Подгонка полинома 4-й степени
    poly_coef = np.polyfit(standard_sizes, matching_peaks, 4)
    size_fit = np.linspace(np.min(standard_sizes), np.max(standard_sizes), 100)
    peak_fit = np.polyval(poly_coef, size_fit)

    return SDFindResult(
        corrected_signal=corrected_signal,
        peaks=matching_peaks,
        peak_areas=peak_areas,
        concentrations=concentrations,
        molarity=molarity,
        size_fit=size_fit,
        peak_fit=peak_fit,
    )
