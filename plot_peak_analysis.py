import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Tuple, Any
import warnings


def plot_peak_analysis(
    data: np.ndarray,
    peaks: Union[np.ndarray, List[int]],
    points: Union[np.ndarray, List[float]],
    selected_peaks_indices: Union[np.ndarray, List[int]],
    sizes: Optional[Union[np.ndarray, List[float]]] = None,
    title: str = "Анализ пиков",
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
    show: bool = False,
    dpi: int = 100,
    highlight_color: str = 'red',
    background_color: str = 'lightyellow'
) -> plt.Figure:
    """
    Визуализирует анализ пиков сигнала электрофореза ДНК с балльной оценкой и аннотациями.
    
    Строит три графика в одной фигуре:
    1. Все пики с баллами (серые = все пики, красные = выбранные по баллам)
    2. Выбранные пики с присвоенными размерами фрагментов (в пн)
    3. Полный обзор со всеми пронумерованными пиками (номер пика + индекс в массиве)
    
    Параметры
    ----------
    data : np.ndarray
        Исходный сигнал (одномерный массив интенсивности).
    peaks : np.ndarray или список целых чисел
        Индексы всех обнаруженных пиков в сигнале.
    points : np.ndarray или список чисел с плавающей точкой
        Баллы качества для каждого пика (соответствуют порядку в `peaks`).
    selected_peaks_indices : np.ndarray или список целых чисел
        Индексы выбранных пиков **в терминах массива `peaks`** (не индексы в `data`!).
        Например: [0, 2, 4] означает выбор 1-го, 3-го и 5-го пиков из `peaks`.
    sizes : np.ndarray, список чисел или None, optional (default=None)
        Размеры фрагментов ДНК (в парах нуклеотидов) для выбранных пиков.
        Длина должна соответствовать количеству выбранных пиков.
    title : str, optional (default="Анализ пиков")
        Заголовок для первого графика и общего заголовка фигуры.
    figsize : tuple[int, int], optional (default=(16, 12))
        Размер фигуры в дюймах (ширина, высота).
    save_path : str или None, optional (default=None)
        Путь для сохранения изображения (например, "peaks_analysis.png").
        Если None — изображение не сохраняется.
    show : bool, optional (default=True)
        Показывать ли график через plt.show().
    dpi : int, optional (default=100)
        Разрешение изображения при сохранении.
    highlight_color : str, optional (default="red")
        Цвет для выделения выбранных пиков.
    background_color : str, optional (default="lightyellow")
        Цвет фона для аннотаций выбранных пиков.
    
    Возвращает
    ----------
    matplotlib.figure.Figure
        Объект фигуры matplotlib для дальнейшей обработки или сохранения.
    
    Примеры
    --------
    >>> # Базовое использование
    >>> fig = plot_peak_analysis(
    ...     data=signal,
    ...     peaks=peak_indices,
    ...     points=peak_scores,
    ...     selected_peaks_indices=[0, 2, 4, 6],
    ...     title="Образец №123"
    ... )
    
    >>> # С присвоенными размерами фрагментов
    >>> fig = plot_peak_analysis(
    ...     data=signal,
    ...     peaks=peak_indices,
    ...     points=peak_scores,
    ...     selected_peaks_indices=[0, 2, 4, 6],
    ...     sizes=[100, 150, 200, 250],  # размеры в пн
    ...     save_path="analysis.png"
    ... )
    
    >>> # Без показа графика (только сохранение)
    >>> fig = plot_peak_analysis(
    ...     data=signal,
    ...     peaks=peak_indices,
    ...     points=peak_scores,
    ...     selected_peaks_indices=[0, 2, 4, 6],
    ...     show=False,
    ...     save_path="batch_output.png"
    ... )
    
    Примечания
    ----------
    1. Индексы в `selected_peaks_indices` относятся к массиву `peaks`, а не к `data`:
       - `peaks[selected_peaks_indices[i]]` — позиция пика в сигнале `data`
    
    2. Цветовая схема:
       - Серые маркеры: все обнаруженные пики
       - Красные маркеры: выбранные пики (топ по баллам)
       - Жёлтые аннотации: баллы для выбранных пиков
       - Светло-серые аннотации: баллы для остальных пиков
    
    3. На втором графике ось X перекалибрована в пары нуклеотидов (если указаны `sizes`).
       Пики сортируются по позиции в сигнале для корректного отображения калибровки.
    
    4. Третий график предназначен для отладки — показывает номер каждого пика
       и его точную позицию в массиве данных (в скобках).
    
    5. Рекомендуемые размеры для публикации:
       - Для статьи: figsize=(12, 8), dpi=300
       - Для презентации: figsize=(16, 9), dpi=150
    """
    # === Валидация входных данных ===
    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=np.float64)
    
    if data.ndim != 1:
        raise ValueError(f"Сигнал должен быть одномерным, получено: {data.ndim}D")
    
    if len(data) < 10:
        raise ValueError("Сигнал слишком короткий (< 10 точек)")
    
    # Конвертация в numpy массивы
    peaks_arr = np.asarray(peaks, dtype=np.int64)
    points_arr = np.asarray(points, dtype=np.float64)
    selected_indices_arr = np.asarray(selected_peaks_indices, dtype=np.int64)
    
    # Проверка длин массивов
    if len(peaks_arr) != len(points_arr):
        raise ValueError(
            f"Длина массивов не совпадает: peaks ({len(peaks_arr)}) != points ({len(points_arr)})"
        )
    
    if len(selected_indices_arr) > len(peaks_arr):
        raise ValueError(
            f"Количество выбранных пиков ({len(selected_indices_arr)}) "
            f"превышает общее число пиков ({len(peaks_arr)})"
        )
    
    # Проверка границ индексов
    if np.any(selected_indices_arr < 0) or np.any(selected_indices_arr >= len(peaks_arr)):
        raise ValueError(
            f"Некоторые индексы в selected_peaks_indices выходят за границы массива peaks "
            f"(допустимый диапазон: 0–{len(peaks_arr)})"
        )
    
    # Проверка размеров (если указаны)
    if sizes is not None:
        sizes_arr = np.asarray(sizes, dtype=np.float64)
        if len(sizes_arr) != len(selected_indices_arr):
            warnings.warn(
                f"Количество размеров ({len(sizes_arr)}) не совпадает с количеством "
                f"выбранных пиков ({len(selected_indices_arr)}). Используются первые {min(len(sizes_arr), len(selected_indices_arr))} размеров.",
                UserWarning
            )
            sizes_arr = sizes_arr[:len(selected_indices_arr)]
    else:
        sizes_arr = None
    
    # Сортировка выбранных пиков по их позиции в сигнале (для корректной калибровки)
    selected_peaks_pos = peaks_arr[selected_indices_arr]
    sort_order = np.argsort(selected_peaks_pos)
    selected_indices_sorted = selected_indices_arr[sort_order]
    selected_peaks_pos_sorted = selected_peaks_pos[sort_order]
    
    if sizes_arr is not None:
        sizes_arr = sizes_arr[sort_order]
    
    n_peaks = len(peaks_arr)
    n_selected = len(selected_indices_arr)
    
    # === Построение графиков ===
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # --- График 1: Все пики с баллами ---
    ax1 = plt.subplot(3, 1, 1)
    x = np.arange(len(data))
    ax1.plot(x, data, linewidth=2, color='blue', label='Сигнал', alpha=0.7)
    
    # Все пики (серые)
    ax1.plot(peaks_arr, data[peaks_arr], 'o', color='gray', markersize=6,
             label=f'Все пики ({n_peaks})', alpha=0.5, zorder=3)
    
    # Выбранные пики (выделенные цветом)
    ax1.plot(selected_peaks_pos_sorted, data[selected_peaks_pos_sorted], 'o',
             color=highlight_color, markersize=10,
             label=f'Выбранные пики ({n_selected})', zorder=5)
    
    # Подписи с баллами
    for i in range(n_peaks):
        is_selected = i in selected_indices_arr
        color = highlight_color if is_selected else 'gray'
        fontweight = 'bold' if is_selected else 'normal'
        fontsize = 9 if is_selected else 7
        bbox_facecolor = background_color if is_selected else 'lightgray'
        
        ax1.text(
            peaks_arr[i], data[peaks_arr[i]] * 1.02,
            f'{points_arr[i]:.1f}',
            verticalalignment='bottom',
            horizontalalignment='center',
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
            bbox=dict(
                boxstyle='round,pad=0.2',
                facecolor=bbox_facecolor,
                alpha=0.7,
                edgecolor=color if is_selected else 'none'
            )
        )
    
    ax1.set_xlabel('Номер точки (позиция)', fontsize=12)
    ax1.set_ylabel('Интенсивность', fontsize=12)
    ax1.set_title(f'{title} - Балльная оценка (выделены топ-{n_selected} пиков)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(0, len(data))
    ax1.set_ylim(0, np.max(data) * 1.15)
    
    # --- График 2: Выбранные пики с размерами ---
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(x, data, linewidth=2, color='blue', label='Сигнал', alpha=0.7)
    
    # Выбранные пики
    ax2.plot(selected_peaks_pos_sorted, data[selected_peaks_pos_sorted], 'o',
             color=highlight_color, markersize=10, zorder=5)
    
    # Подписи с размерами
    if sizes_arr is not None:
        for i, (peak_pos, size_val) in enumerate(zip(selected_peaks_pos_sorted, sizes_arr)):
            ax2.text(
                peak_pos, data[peak_pos] * 1.02,
                f'{size_val:.0f} п.н.',
                verticalalignment='bottom',
                horizontalalignment='center',
                fontsize=10,
                fontweight='bold',
                color='darkgreen',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='lightgreen',
                    alpha=0.8
                )
            )
    else:
        # Если размеры не указаны — показываем только позиции
        for i, peak_pos in enumerate(selected_peaks_pos_sorted):
            ax2.text(
                peak_pos, data[peak_pos] * 1.02,
                f'Поз. {peak_pos}',
                verticalalignment='bottom',
                horizontalalignment='center',
                fontsize=9,
                fontweight='bold',
                color='darkgreen',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='lightgreen',
                    alpha=0.7
                )
            )
    
    ax2.set_xlabel('Позиция в сигнале', fontsize=12)
    ax2.set_ylabel('Интенсивность', fontsize=12)
    ax2.set_title(
        f'Выбранные топ-{n_selected} пиков ' +
        ('с размерами фрагментов' if sizes_arr is not None else 'без калибровки'),
        fontsize=14, fontweight='bold'
    )
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(0, len(data))
    ax2.set_ylim(0, np.max(data) * 1.15)
    
    # --- График 3: Полный обзор со всеми пронумерованными пиками ---
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(x, data, linewidth=2, color='blue', alpha=0.7, label='Сигнал')
    
    # Все пики
    ax3.plot(peaks_arr, data[peaks_arr], 'o', color='gray', markersize=8,
             alpha=0.6, zorder=3, label=f'Все пики ({n_peaks})')
    
    # Выбранные пики
    ax3.plot(selected_peaks_pos_sorted, data[selected_peaks_pos_sorted], 'o',
             color=highlight_color, markersize=10, zorder=5,
             label=f'Выбранные ({n_selected})')
    
    # Нумерация всех пиков
    for i in range(n_peaks):
        peak_pos = peaks_arr[i]
        is_selected = i in selected_indices_arr
        
        color = highlight_color if is_selected else 'blue'
        fontweight = 'bold' if is_selected else 'normal'
        fontsize = 9 if is_selected else 7
        facecolor = background_color if is_selected else 'lightblue'
        
        ax3.text(
            peak_pos, data[peak_pos] * 1.03,
            f'#{i + 1}\n[{peak_pos}]',
            verticalalignment='bottom',
            horizontalalignment='center',
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor=facecolor,
                alpha=0.7,
                edgecolor=color
            )
        )
    
    ax3.set_xlabel('Индекс в массиве данных', fontsize=12)
    ax3.set_ylabel('Интенсивность', fontsize=12)
    ax3.set_title('Полный обзор: все пики пронумерованы (#номер_пика / [индекс_в_данных])',
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_xlim(0, len(data))
    ax3.set_ylim(0, np.max(data) * 1.15)
    
    # Общий заголовок фигуры
    fig.suptitle(f'Анализ пиков: {title}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Оставляем место для общего заголовка
    
    # Сохранение в файл
    if save_path is not None:
        try:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✅ График сохранён: {save_path}")
        except Exception as e:
            warnings.warn(f"Ошибка при сохранении графика: {e}", UserWarning)
    
    # Показ графика
    if show:
        plt.show()
    
    return fig


# === Пример использования ===
if __name__ == "__main__":
    # Генерация тестового сигнала с пиками
    np.random.seed(42)
    x = np.linspace(0, 10, 1000)
    signal = np.sin(x * 5) * 10 + np.random.randn(1000) * 0.5
    
    # Добавление "пиков" ДНК-фрагментов
    peak_positions = [150, 300, 450, 600, 750, 850]
    for pos in peak_positions:
        signal[pos-15:pos+15] += np.exp(-((np.arange(30) - 15) ** 2) / 30) * 40
    
    # Обнаружение пиков
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(signal, height=20, distance=50)
    
    # Генерация баллов (имитация алгоритма оценки)
    np.random.seed(0)
    scores = np.random.uniform(1.0, 5.0, size=len(peaks))
    scores[::2] += 2.0  # Некоторые пики получают более высокие баллы
    
    # Выбор топ-3 пиков по баллам
    top_indices = np.argsort(scores)[-3:][::-1]  # Индексы в массиве peaks
    
    # Размеры фрагментов (в пн) для выбранных пиков
    fragment_sizes = [100, 250, 400]
    
    # Построение графиков
    fig = plot_peak_analysis(
        data=signal,
        peaks=peaks,
        points=scores,
        selected_peaks_indices=top_indices,
        sizes=fragment_sizes,
        title="Образец ДНК #A-2024-001",
        figsize=(14, 10),
        save_path="peak_analysis_example.png",
        dpi=150
    )
    
    print("✅ Анализ завершён. График сохранён в 'peak_analysis_example.png'")