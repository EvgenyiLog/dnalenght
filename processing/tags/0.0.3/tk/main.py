from tkinter.filedialog import askopenfilenames
from xml.etree import ElementTree
from xml.etree.ElementTree import Element
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.pyplot import subplots
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sdfind import SDFind
from glfind import GLFind
from typing import Any
import numpy as np


def analyze_frf_files() -> None:
    filenames = askopenfilenames(
        title="Выберите файлы .frf",
        filetypes=[("FRF files", "*.frf")],
    )

    if not filenames:
        print("Файлы не были выбраны.")
        return

    # Инициализация массивов для хранения данных
    SizeStandard_titles: list[str] = []
    SizeStandard_data: list[list[float]] = []
    Sizes: list[list[float]] = []
    Concentrations: list[list[float]] = []
    ReleaseTimes: list[list[int]] = []

    GenLib_titles: list[str] = []
    GenLib_data: list[list[float]] = []

    for filename in filenames:
        try:
            file = open(filename, encoding='utf-8')
            root = ElementTree.parse(file)
            raw_title: str = root.findtext('Title')  # type: ignore
            type_value = root.findtext('Type')
            if type_value == 'AllelicLadder':
                size_standard_node = root.find('./SizeStandard/Sizes')
                if size_standard_node is not None:
                    sizes: list[float] = []
                    concentrations: list[float] = []
                    release_times: list[int] = []

                    size_elements = size_standard_node.findall('double')
                    for size_elem in size_elements:
                        sizes.append(float(size_elem.text))  # type: ignore
                        concentrations.append(float(size_elem.get('Concentration')))  # type: ignore
                        release_time_str: str = size_elem.get('ReleaseTime')  # type: ignore
                        time_parts = release_time_str.split(':')
                        release_times.append(int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2]))

                    Sizes.append(sizes)
                    Concentrations.append(concentrations)
                    ReleaseTimes.append(release_times)
                    SizeStandard_titles.append(raw_title)
                    SizeStandard_data.append(extract_filtered_int_values(root.findall('./Data/Point')))
            elif type_value == 'Sample':
                GenLib_titles.append('GenLib_' + raw_title)
                GenLib_data.append(extract_filtered_int_values(root.findall('./Data/Point')))
        except Exception as ex:
            print(ex)

    create_gui(SizeStandard_titles, SizeStandard_data, Sizes, Concentrations, ReleaseTimes, GenLib_titles, GenLib_data)


def extract_filtered_int_values(point_nodes: list[Element]) -> list[float]:
    values: list[float] = []
    for point in point_nodes:
        data_node = point.find('Data')
        if data_node is not None:
            int_nodes = data_node.findall('int')
            for node in int_nodes:
                val = float(node.text)  # type: ignore
                if val != 1:
                    values.append(val)
    return values


def create_gui(SizeStandard_titles: list[str], SizeStandard_data: list[list[float]], Sizes: list[list[float]], Concentrations: list[list[float]], ReleaseTimes: list[list[int]], GenLib_titles: list[str], GenLib_data: list[list[float]]) -> None:
    root = tk.Tk()

    def quit_me():
        print('quit')
        root.quit()
        root.destroy()

    root.title("Обработка файлов FRF")
    root.geometry("1200x800")
    root.rowconfigure(1, weight=1)
    root.rowconfigure(3, weight=1)
    root.columnconfigure(1, weight=1)

    tk.Label(root, text="Стандарты длин", font=("Arial", 14, "bold")).grid(row=0, columnspan=2)

    # Окно для графика стандартов длин
    fig_standard, ax_standard = subplots()
    canvas_standard = FigureCanvasTkAgg(fig_standard, master=root)
    canvas_standard.get_tk_widget().grid(row=1, column=1, sticky=tk.NSEW)

    def plot_standard() -> None:
        i = standard_variable.get()
        if i >= len(SizeStandard_data):
            return
        ax_standard.clear()
        ax_standard.plot(SizeStandard_data[i])
        ax_standard.set_title(SizeStandard_titles[i])
        canvas_standard.draw()

    standard_picker = tk.Frame(root)
    standard_picker.grid(row=1, column=0, sticky=tk.NSEW)
    standard_variable = tk.IntVar(standard_picker, 0)
    for i, title in enumerate(SizeStandard_titles):
        tk.Radiobutton(standard_picker, text=title, value=i, variable=standard_variable, command=plot_standard).pack(padx=(4, 4), pady=(4, 4), anchor=tk.NW)

    tk.Label(root, text="Геномные библиотеки", font=("Arial", 14, "bold")).grid(row=2, columnspan=2)

    # Окно для графика геномных библиотек
    fig_genlib, ax_genlib = subplots()
    canvas_genlib = FigureCanvasTkAgg(fig_genlib, master=root)
    canvas_genlib.get_tk_widget().grid(row=3, column=1, sticky=tk.NSEW)

    def plot_genlib(i) -> None:
        ax_genlib.clear()
        ax_genlib.plot(GenLib_data[i])
        ax_genlib.set_title(GenLib_titles[i])
        canvas_genlib.draw()

    def genlib_select_all() -> None:
        val = genlib_variable_select_all.get()
        for v in genlib_variables:
            v.set(val)

    def genlib_select_one() -> None:
        val = 1 if all([v.get() == 1 for v in genlib_variables]) else 0
        genlib_variable_select_all.set(val)

    genlib_picker = tk.Frame(root)
    genlib_picker.columnconfigure(1, weight=1)
    genlib_picker.grid(row=3, column=0, sticky=tk.NSEW)
    genlib_variables: list[tk.IntVar] = []
    genlib_variable_select_all = tk.IntVar(genlib_picker, 0)
    tk.Checkbutton(genlib_picker, text='Выбрать все', command=genlib_select_all, variable=genlib_variable_select_all).grid(row=0, column=0, columnspan=2, padx=(4, 4), pady=(4, 4), sticky=tk.NW)
    for i, title in enumerate(GenLib_titles):
        genlib_variable = tk.IntVar(genlib_picker, 0)
        tk.Checkbutton(genlib_picker, command=genlib_select_one, variable=genlib_variable).grid(row=i + 1, column=0, padx=(4, 0), pady=(4, 4))

        def cmd(i: int = i):
            plot_genlib(i)
        tk.Button(genlib_picker, text=title, command=cmd).grid(row=i + 1, column=1, padx=(0, 4), pady=(4, 4))
        genlib_variables.append(genlib_variable)

    def analyze_data() -> None:
        selected_standard_idx = standard_variable.get()

        if selected_standard_idx >= len(SizeStandard_data):
            messagebox.showerror('Ошибка', 'Нет данных для выбранного стандарта длины.')
            return

        title: str = SizeStandard_titles[selected_standard_idx]
        data: list[float] = SizeStandard_data[selected_standard_idx]
        sizes: list[float] = Sizes[selected_standard_idx]
        concentrations: list[float] = Concentrations[selected_standard_idx]
        release_times: list[int] = ReleaseTimes[selected_standard_idx]

        genlib_titles: list[str] = []
        genlib_data: list[list[float]] = []

        for i, v in enumerate(genlib_variables):
            if v.get() == 1:
                genlib_titles.append(GenLib_titles[i])
                genlib_data.append(GenLib_data[i])

        if len(genlib_data) == 0:
            messagebox.showerror('Ошибка', 'Выберите хотя бы одну геномную библиотеку!')
            return

        [peak, led_area, led_conc, ZrRef, SD_molarity] = SDFind(data, sizes, release_times, concentrations)
        print(peak, led_area, led_conc, ZrRef, SD_molarity)

        results: list[dict[str, Any]] = []
        for gl_d in genlib_data:
            (
                t_main, denoised_data, st_peaks, st_length, t_unrecognized_peaks, unrecognized_peaks,
                lib_length, lib_peak_locations, t_final_locations, final_filtered_below_threshold_locations,
                hpx, unr, stp, main_corr, gl_areas, peaks_corr, library_peaks, area_corr, molarity,
                max_lib_peak, max_lib_value, total_lib_area, total_lib_conc, total_lib_molarity,
                x_fill, y_fill, x_lib_fill, y_lib_fill
            ) = GLFind(gl_d, peak, sizes, concentrations)
            results.append({
                't_main': t_main,
                'denoised_data': denoised_data,
                'st_peaks': st_peaks,
                'st_length': st_length,
                't_unrecognized_peaks': t_unrecognized_peaks,
                'unrecognized_peaks': unrecognized_peaks,
                'lib_length': lib_length,
                'LibPeakLocations': lib_peak_locations,
                't_final_locations': t_final_locations,
                'final_filtered_below_threshold_locations': final_filtered_below_threshold_locations,
                'hpx': hpx,
                'unr': unr,
                'stp': stp,
                'mainCorr': main_corr,
                'GLAreas': gl_areas,
                'peaksCorr': peaks_corr,
                'library_peaks': library_peaks,
                'areaCorr': area_corr,
                'molarity': molarity,
                'maxLibPeak': max_lib_peak,
                'maxLibValue': max_lib_value,
                'totalLibArea': total_lib_area,
                'totalLibConc': total_lib_conc,
                'totalLibMolarity': total_lib_molarity,
                'x_fill': x_fill,
                'y_fill': y_fill,
                'x_Lib_fill': x_lib_fill,
                'y_Lib_fill': y_lib_fill,
            })
        show_results(
            title, genlib_titles, genlib_data, ZrRef, peak, sizes, results, concentrations, led_area, peaks_corr, library_peaks, area_corr, gl_areas, molarity, SD_molarity, max_lib_peak, max_lib_value, total_lib_area, total_lib_conc, total_lib_molarity, x_fill, y_fill, x_lib_fill, y_lib_fill
        )

    tk.Button(root, text="Анализ", font=("Arial", 14, "bold"), command=analyze_data).grid(row=4, column=1, padx=(4, 4), pady=(4, 4), sticky=tk.E)

    plot_standard()
    root.mainloop()


def show_results(title, genlib_titles, genlib_data, ZrRef, peak, sizes, results, concentrations, led_area, peaks_corr, library_peaks, area_corr, gl_areas, molarity, SD_molarity, max_lib_peak, max_lib_value, total_lib_area, total_lib_conc, total_lib_molarity, x_fill, y_fill, x_lib_fill, y_lib_fill) -> None:
    root = tk.Toplevel()
    root.geometry("1200x800")

    root.rowconfigure(1, weight=1)
    root.rowconfigure(4, weight=1)
    root.columnconfigure(1, weight=1)

    tk.Label(root, text="Стандарты длин", font=("Arial", 14, "bold")).grid(row=0, columnspan=2)

    # Окно для графика стандартов длин
    fig_standard, ax_standard = subplots()
    canvas_standard = FigureCanvasTkAgg(fig_standard, master=root)
    canvas_standard.get_tk_widget().grid(row=1, column=1, sticky=tk.NSEW)

    def plot_size_standard() -> None:
        # График анализа стандарта длин
        pks = ZrRef[peak]
        ax_standard.clear()
        ax_standard.plot(ZrRef, linewidth=2)

        # Добавление рисок и значений на ось x
        ax_standard.set_xticks(peak)
        ax_standard.set_xticklabels(sizes, fontsize=16)

        # Отметка пиков
        ax_standard.stem(peak, pks, linefmt='r', markerfmt='ro', basefmt=" ")
        for i in range(len(peak)):
            ax_standard.text(peak[i], pks[i] + 0.01 * max(pks), f'{sizes[i]}', verticalalignment='bottom', horizontalalignment='right', fontsize=18)

        ax_standard.set_xlabel('Длина фрагментов, пн', fontsize=16)
        ax_standard.set_ylabel('Интенсивность', fontsize=16)
        ax_standard.set_title(title)
        ax_standard.grid(True)
        canvas_standard.draw()

    def show_standard_table() -> None:
        root = tk.Toplevel()
        root.title(f"Стандарт длин: {title}")
        root.geometry("1200x800")
        root.rowconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)
        fig, ax = subplots()
        ax.scatter(sizes, peak, label="Исходные данные", color='blue')

        # Подгонка полинома 4-й степени
        p = np.polyfit(sizes, peak, 4)
        liz_fit = np.linspace(np.min(sizes), np.max(sizes), 100)
        locs_fit = np.polyval(p, liz_fit)

        ax.plot(liz_fit, locs_fit, 'r-', linewidth=2, label="Подгонка полинома")
        ax.set_xlabel("Длина фрагментов, пн")
        ax.set_ylabel("Время выхода, с")
        ax.set_title("Калибровочная кривая")
        ax.grid(True)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)
        canvas.draw()

        columns = ("Длина фрагментов, пн", "Концентрация, нг/мкл", "Молярность, нмоль/л", "Время выхода, с", "Площадь * 10^7")
        tree = ttk.Treeview(root, columns=columns, show='headings')

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor=tk.CENTER)

        # Round data and insert into table
        for i in range(len(sizes)):
            tree.insert('', tk.END, values=[
                format_value(np.round(sizes[i])),
                format_value(concentrations[i]),
                format_value(SD_molarity[i]),
                format_value(np.round(peak[i])),
                format_value(np.round(led_area[i] * 100) / 100)
            ])

        tree.grid(row=1, column=0, sticky=tk.NSEW)

    tk.Button(root, text="Отчет", font=("Arial", 14, "bold"), command=show_standard_table).grid(row=2, column=1, padx=(4, 4), pady=(4, 4), sticky=tk.E)

    tk.Label(root, text="Геномные библиотеки", font=("Arial", 14, "bold")).grid(row=3, columnspan=2)

    # Окно для графика геномных библиотек
    fig_genlib, ax_genlib = subplots()
    canvas_genlib = FigureCanvasTkAgg(fig_genlib, master=root)
    canvas_genlib.get_tk_widget().grid(row=4, column=1, sticky=tk.NSEW)

    genlib_picker = tk.Frame(root)
    genlib_picker.grid(row=4, column=0, sticky=tk.NSEW)
    genlib_variable = tk.IntVar(genlib_picker, 0)

    def plot_genlib() -> None:
        selected_idx = genlib_variable.get()
        current_result = results[selected_idx]

        t_main = current_result['t_main']
        denoised_data = current_result['denoised_data']
        st_peaks = current_result['st_peaks']
        st_length = current_result['st_length']
        t_unrecognized_peaks = current_result['t_unrecognized_peaks']
        unrecognized_peaks = current_result['unrecognized_peaks']
        lib_length = current_result['lib_length']
        lib_peak_locations = current_result['LibPeakLocations']
        t_final_locations = current_result['t_final_locations']
        hpx = current_result['hpx']
        unr = current_result['unr']
        stp = current_result['stp']
        x_fill = current_result['x_fill']
        y_fill = current_result['y_fill']
        x_lib_fill = current_result['x_Lib_fill']
        y_lib_fill = current_result['y_Lib_fill']

        print(f'unrecognized: {unrecognized_peaks}')

        ax_genlib.clear()
        ax_genlib.plot(t_main, denoised_data, linewidth=2)
        ax_genlib.scatter(st_peaks, denoised_data[np.array(st_length).astype(int)], color='red', marker='x', s=100, linewidths=2)
        ax_genlib.scatter(t_unrecognized_peaks, denoised_data[np.array(unrecognized_peaks).astype(int)], color='blue', marker='*', s=100, linewidths=2)
        ax_genlib.scatter(lib_length, denoised_data[np.array(lib_peak_locations).astype(int)], color='red', edgecolors='black', s=90)

        if len(x_fill) > 0:
            ax_genlib.fill_between(x_fill, y_fill, alpha=0.3, color='red')

        if len(x_lib_fill) > 0:
            ax_genlib.fill_between(x_lib_fill, y_lib_fill, alpha=0.5, color='red')

        for i in range(len(t_final_locations)):
            ax_genlib.axvline(t_final_locations[i], linestyle='--', color='blue')

        ax_genlib.axhline(0, linestyle='--', color='black')

        for i in range(len(st_peaks)):
            ax_genlib.text(st_peaks[i], denoised_data[np.array(st_length).astype(int)[i]] + 0.01 * max(denoised_data),
                           f'{stp[i]}', fontsize=18, verticalalignment='bottom', horizontalalignment='right')

        for i in range(len(hpx)):
            ax_genlib.text(lib_length[i], denoised_data[np.array(lib_peak_locations).astype(int)[i]] + 0.05 * max(denoised_data),
                           f'{hpx[i]}', fontsize=13, verticalalignment='bottom', horizontalalignment='right', rotation=90)

        for i in range(len(unr)):
            ax_genlib.text(t_unrecognized_peaks[i], denoised_data[np.array(unrecognized_peaks).astype(int)[i]] + 0.01 * max(denoised_data),
                           f'{unr[i]}', fontsize=18, verticalalignment='bottom', horizontalalignment='right')

        ax_genlib.set_xlabel('Длина фрагментов, пн', fontsize=16)
        ax_genlib.set_ylabel('Интенсивность', fontsize=16)
        ax_genlib.set_title(genlib_titles[selected_idx])
        ax_genlib.grid(True)
        canvas_genlib.draw()

    for i, title in enumerate(genlib_titles):
        tk.Radiobutton(genlib_picker, text=title, value=i, variable=genlib_variable, command=plot_genlib).pack(padx=(4, 4), pady=(4, 4), anchor=tk.NW)

    def show_genlib_table():
        selected_idx = genlib_variable.get()
        root = tk.Toplevel()
        root.title(f"Геномная библиотека: {genlib_titles[selected_idx]}")
        root.geometry("1200x800")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        current_result = results[selected_idx]
        area_corr = current_result['areaCorr']
        molarity = current_result['molarity']
        peaks_corr = np.round(current_result['peaksCorr'])
        library_peaks = np.round(current_result['library_peaks'])
        gl_areas = np.round(current_result['GLAreas'] * 100) / 100

        if len(peaks_corr) == 0 or len(library_peaks) == 0:
            messagebox.showerror('Ошибка', 'Нет данных для выбранной библиотеки.')
            return

        columns1 = ("Длина фрагментов, пн", "Концентрация, нг/мкл", "Молярность, нмоль/л", "Время выхода, с", "Площадь * 10^7")

        tree1 = ttk.Treeview(root, columns=columns1, show="headings")

        for col in columns1:
            tree1.heading(col, text=col)
            tree1.column(col, anchor=tk.CENTER)

        for i in range(len(peaks_corr)):
            tree1.insert("", tk.END, values=[
                format_value(peaks_corr[i]),
                format_value(area_corr[i]),
                format_value(molarity[i]),
                format_value(library_peaks[i]),
                format_value(gl_areas[i])
            ])

        tree1.grid(row=0, sticky=tk.NSEW)

        max_lib_peak = np.round(current_result['maxLibPeak'])
        max_lib_value = np.round(current_result['maxLibValue'])
        total_lib_area = np.round(current_result['totalLibArea'] * 100) / 100
        total_lib_conc = np.round(current_result['totalLibConc'] * 100) / 100
        total_lib_molarity = np.round(current_result['totalLibMolarity'] * 100) / 100

        columns2 = ('Длина максимального фрагмента, пн', 'Концентрация геномной библиотеки, нг/мкл', 'Молярность геномной библиотеки, пмоль/л', 'Время выхода максимального фрагмента, с', 'Площадь геномной библиотеки * 10^7')

        tree2 = ttk.Treeview(root, columns=columns2, show="headings")
        tree2.grid(row=1, sticky=tk.NSEW)

        for col in columns2:
            tree2.heading(col, text=col)
            tree2.column(col, anchor=tk.CENTER)

        tree2.insert("", tk.END, values=[
            format_value(max_lib_peak),
            format_value(total_lib_conc),
            format_value(total_lib_molarity),
            format_value(max_lib_value),
            format_value(total_lib_area)
        ])

    tk.Button(root, text="Отчет", font=("Arial", 14, "bold"), command=show_genlib_table).grid(row=5, column=1, padx=(4, 4), pady=(4, 4), sticky=tk.E)

    plot_size_standard()
    plot_genlib()


def format_value(x: Any) -> str:
    if x % 1 == 0:
        return f"{int(x)}"
    elif round(x * 10) == x * 10:
        return f"{x:.1f}"
    else:
        return f"{x:.2f}"


if __name__ == "__main__":
    analyze_frf_files()
