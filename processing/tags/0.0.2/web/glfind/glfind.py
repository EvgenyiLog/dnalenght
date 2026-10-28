from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from lib.glfind.classify_and_extract_library_peaks import classify_and_extract_library_peaks
from lib.glfind.compute_fragmented_library_concentrations import compute_fragmented_library_concentrations
from lib.glfind.compute_smooth_library_concentrations import compute_smooth_library_concentrations
from lib.glfind.filter_isolated_peaks import filter_isolated_peaks
from lib.glfind.select_reference_peaks import select_reference_peaks
from lib.glfind.find_significant_peaks import find_significant_peaks
from lib.glfind.find_signal_minima import find_signal_minima
from lib.glfind.handle_smooth_library_case import handle_smooth_library_case
from lib.glfind.refine_library_peaks import refine_library_peaks
from lib.glfind.refine_minima_near_reference_peaks import refine_minima_near_reference_peaks
from lib.glfind.select_isolated_peaks import select_isolated_peaks
from lib.matlab.msbackadj import msbackadj
from lib.matlab.round import matlab_round


@dataclass
class GLFindResult:
    t_main: NDArray[np.floating]
    corrected_data: NDArray[np.floating]
    st_peaks: NDArray[np.integer]
    st_length: NDArray[np.integer]
    t_unrecognized_peaks: NDArray[np.floating]
    unrecognized_peaks: NDArray[np.integer]
    lib_length: NDArray[np.floating]
    lib_peak_locations: NDArray[np.integer]
    t_final_locations: NDArray[np.floating]
    final_lib_local_minimums: NDArray[np.integer]
    hpx: NDArray[np.integer]
    unr: NDArray[np.integer]
    stp: NDArray[np.integer]
    main_corr: NDArray[np.floating]
    all_areas: NDArray[np.floating]
    all_peaks_corr: NDArray[np.floating]
    all_peaks: NDArray[np.integer]
    all_areas_conc: NDArray[np.floating]
    molarity: NDArray[np.floating]
    max_lib_peak: np.floating
    max_lib_value: np.integer
    total_lib_area: np.floating
    total_lib_conc: np.floating
    total_lib_molarity: np.floating
    x_fill: NDArray[np.floating]
    y_fill: NDArray[np.floating]
    x_lib_fill: NDArray[np.floating]
    y_lib_fill: NDArray[np.floating]


def glfind(
    raw_signal: NDArray[np.floating],
    standard_peaks: NDArray[np.integer],
    standard_sizes: NDArray[np.floating],
    standard_conc: NDArray[np.floating],
) -> GLFindResult:
    # 1. Выбор первых 50 значений как шума
    noise = raw_signal[:50]
    # Вычитание шума из данных
    denoised_signal = raw_signal - np.mean(noise)
    x = np.arange(len(raw_signal))
    corrected_signal = msbackadj(x, denoised_signal, window_size=140, step_size=300, quantile_value=0.05)  # коррекция бейзлайна

    significant_peak_candidates = find_significant_peaks(corrected_signal)

    if len(significant_peak_candidates) == 0:
        raise ValueError('Пики геномной библиотеки не были найдены')

    isolated_peaks_candidates, significant_peaks = select_isolated_peaks(significant_peak_candidates, corrected_signal)

    isolated_peaks = filter_isolated_peaks(isolated_peaks_candidates, significant_peaks, corrected_signal)

    # Вычисление pace
    pace: np.int64 = standard_peaks[-1] - standard_peaks[0]
    reference_peaks, pre_unrecognized_peaks = select_reference_peaks(isolated_peaks, pace, corrected_signal)

    if len(reference_peaks) != 2:
        # TODO Добавить выбор реперных пиков
        # Мы обрабатывает несколько генных библиотек в одном анализе и для каждой может потребоваться свой выбор
        raise ValueError('Реперные пики не найдены.')

    # Удаляем пики, которые лежат за пределами реперов
    significant_peaks = significant_peaks[(significant_peaks >= reference_peaks[0]) & (significant_peaks <= reference_peaks[-1])]

    minima_candidates, all_local_minimums = find_signal_minima(corrected_signal)

    complete_peaks_locations = refine_minima_near_reference_peaks(minima_candidates, reference_peaks, pre_unrecognized_peaks, corrected_signal)

    # 4. Калибровка данных
    sdc = np.polyfit(standard_peaks, standard_sizes, 5)  # калибровка по стандарту
    sdc2 = np.polyfit(standard_sizes, standard_peaks, 5)  # калибровка в обратную сторону для проверки соотвествия границ

    # 5. Обработка данных с учётом калибровки
    # В этом блоке теперь находим и разбиваем все локальные пики по классам: реперные пики, пики геномной библиотеки и неопознанные пики
    (
        library_peak_candidates,
        hidden_lib_peak_locations,
        hidden_lib_areas,
        lib_local_minima_candidates,
        unrecognized_peaks,
        max_lib_value,

        st_areas,
        x_fill_1,
        x_lib_fill_1,
        y_fill,
        y_lib_fill,
    ) = classify_and_extract_library_peaks(
        corrected_signal,
        significant_peaks,
        reference_peaks,
        complete_peaks_locations,
        pre_unrecognized_peaks,
        sdc,
        sdc2,
    )

    is_smooth = len(hidden_lib_peak_locations) == 0

    # Если ГБ гладкая/фаикс/слишком низкая
    if is_smooth:
        (
            library_peaks,
            hidden_lib_peak_locations,
            new_hidden_lib_areas,
            final_lib_local_minimums,
            unrecognized_peaks,
            max_lib_value,
        ) = handle_smooth_library_case(
            corrected_signal,
            significant_peaks,
            reference_peaks,
            complete_peaks_locations,
            all_local_minimums,
        )
        hidden_lib_areas = np.append(hidden_lib_areas, new_hidden_lib_areas)
    else:
        library_peaks, final_lib_local_minimums = refine_library_peaks(
            library_peak_candidates,
            hidden_lib_peak_locations,
            lib_local_minima_candidates,
        )

    # Этот блок приводит электрофореграмму геномной библиотеки и стандарта
    # длин в одну шкалу (выравнивает по ширине и высоте)
    st_peaks = np.array([standard_peaks[0], standard_peaks[-1]], dtype=np.int64)
    px = np.polyfit(reference_peaks, st_peaks, 1)  # выравнивание по ширине
    t = np.arange(len(corrected_signal))
    t_main = np.polyval(px, t)

    # Подсчёт концентраций и молярности по реперам (ГБ будет дальше)
    main_corr = np.polyval(sdc, t_main)
    st_peaks_corr = np.polyval(sdc, st_peaks)

    st_areas = np.array([st_areas[0], st_areas[-1]], dtype=np.float64)
    conc = np.array([standard_conc[0], standard_conc[-1]], dtype=np.float64)

    led_one_area = st_areas / (st_peaks_corr / 100)  # считает корректно, проверено (в Matlab)
    a = np.polyfit(led_one_area, conc, 1)

    st_molarity = ((conc * 1e-3) / (649 * st_peaks_corr)) * 1e9

    hid_lib_length = np.polyval(px, hidden_lib_peak_locations)  # пересчёт по времени
    hid_lib_peaks_corr = np.polyval(sdc, hid_lib_length)

    hid_one_area = hidden_lib_areas / (hid_lib_peaks_corr / 100)  # пересчёт по длине
    hid_one_area_conc = np.polyval(a, hid_one_area)  # находим концентрацию в нг/мкл
    hid_molarity = ((hid_one_area_conc * 1e-3) / (649 * hid_lib_peaks_corr)) * 1e9  # в нмолях/л!

    #  Если ГБ гладкая/фаикс/слишком низкая (потому что там всего один пик - нет локальных)
    if is_smooth:
        (
            lib_areas,
            lib_one_area,
            lib_one_area_conc,
            lib_molarity,
        ) = compute_smooth_library_concentrations(
            hidden_lib_areas,
            hid_one_area,
            hid_one_area_conc,
            hid_molarity,
        )
    # Если ГБ содержит локальные пики и она была идентифицирована как ГБ (был найдет максимум и скрытые пики):
    else:
        (
            lib_areas,
            lib_one_area,
            lib_one_area_conc,
            lib_molarity,
        ) = compute_fragmented_library_concentrations(
            library_peaks,
            hidden_lib_peak_locations,
            hidden_lib_areas,
            hid_one_area,
            hid_one_area_conc,
            hid_molarity,
        )

    # считаем концентрации ГБ
    lib_length = np.polyval(px, library_peaks)
    lib_peaks_corr = np.polyval(sdc, lib_length)

    # one_area = np.concatenate(([led_one_area[0]], lib_one_area, [led_one_area[-1]]))  # площадь на один фрагмент, не нужен в коде, но может понадобиться для проверки!!!
    all_areas_conc = np.concatenate(([standard_conc[0]], lib_one_area_conc, [standard_conc[-1]]))  # концентрация
    all_areas = np.concatenate(([st_areas[0]], lib_areas, [st_areas[-1]]))  # общая площадь фрагмента
    molarity = np.concatenate(([st_molarity[0]], lib_molarity, [st_molarity[-1]]))  # молярность

    all_peaks = np.concatenate(([reference_peaks[0]], library_peaks, [reference_peaks[-1]]))  # время выхода
    all_peaks_corr = np.concatenate(([st_peaks_corr[0]], lib_peaks_corr, [st_peaks_corr[-1]]))  # длина в пн

    t_final_locations = np.polyval(px, final_lib_local_minimums)
    t_unrecognized_peaks = np.polyval(px, unrecognized_peaks)  # пересчёт по времени неизвестных пиков
    unrecognized_peaks_corr = np.polyval(sdc, t_unrecognized_peaks)  # только неопознанные пики
    max_lib_peak = main_corr[max_lib_value]  # максимальный пик библиотеки

    total_lib_area = np.sum(lib_areas)
    total_lib_conc = np.sum(lib_one_area_conc)
    total_lib_molarity = np.sum(lib_molarity)

    # Закрашиваем красным ложные пики и широкую библиотеку
    if len(x_fill_1):
        x_fill_1 = np.array([x_fill_1[0], x_fill_1[-1]])
        x_fill_1 = t_main[x_fill_1.astype(int)]
        x_fill = np.linspace(x_fill_1[0], x_fill_1[-1], 100)
    else:
        x_fill = np.empty(0, dtype=np.float64)

    if len(x_lib_fill_1):
        x_lib_fill_1 = np.array([x_lib_fill_1[0], x_lib_fill_1[-1]])
        x_lib_fill_1 = t_main[x_lib_fill_1.astype(int)]
        x_lib_fill = np.linspace(x_lib_fill_1[0], x_lib_fill_1[-1], 100)
    else:
        x_lib_fill = np.empty(0, dtype=np.float64)

    hpx = matlab_round(lib_peaks_corr)
    unr = matlab_round(unrecognized_peaks_corr)
    stp = matlab_round([standard_sizes[0], standard_sizes[-1]])

    return GLFindResult(
        t_main=t_main,
        corrected_data=corrected_signal,

        st_peaks=st_peaks,
        st_length=reference_peaks,
        stp=stp,

        t_unrecognized_peaks=t_unrecognized_peaks,
        unrecognized_peaks=unrecognized_peaks,
        unr=unr,

        lib_length=lib_length,
        lib_peak_locations=library_peaks,
        hpx=hpx,

        t_final_locations=t_final_locations,
        final_lib_local_minimums=final_lib_local_minimums,
        main_corr=main_corr,

        all_areas=all_areas,
        all_peaks_corr=all_peaks_corr,
        all_peaks=all_peaks,
        all_areas_conc=all_areas_conc,
        molarity=molarity,

        max_lib_peak=max_lib_peak,
        max_lib_value=max_lib_value,
        total_lib_area=total_lib_area,
        total_lib_conc=total_lib_conc,
        total_lib_molarity=total_lib_molarity,

        x_fill=x_fill,
        y_fill=y_fill,

        x_lib_fill=x_lib_fill,
        y_lib_fill=y_lib_fill,
    )
