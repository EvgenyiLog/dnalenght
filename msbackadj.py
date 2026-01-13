import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, PchipInterpolator
from typing import Union, Optional, Callable

def msbackadj(
    X: np.ndarray, 
    Y: np.ndarray, 
    step_size: Union[int, float, Callable] = 200,
    window_size: Union[int, float, Callable] = 200,
    regression_method: str = 'pchip',
    estimation_method: str = 'quantile',
    quantile_value: float = 0.1,
    preserve_heights: bool = False,
    show_plot: Optional[int] = None
) -> np.ndarray:
    """
    Корректировка базовой линии спектра методом оценки локальных минимумов в окнах.

    Args:
        X: Массив значений (m/z или время). Вектор (N,) или матрица (N, M).
        Y: Массив интенсивностей. Вектор (N,) или матрица (N, M).
        step_size: Расстояние между центрами окон (число или функция).
        window_size: Ширина окна для поиска фона (число или функция).
        regression_method: Метод аппроксимации ('pchip', 'linear', 'cubic', 'spline').
        estimation_method: Метод оценки фона в окне ('quantile' или 'mean').
        quantile_value: Значение квантиля (0.1 по умолчанию).
        preserve_heights: Флаг для сохранения относительных высот пиков.
        show_plot: Индекс сигнала для отрисовки графика (None — не рисовать).

    Returns:
        np.ndarray: Скорректированные данные Y (такой же размерности, как входные).
    """
    
    # Приводим к 2D (сигналы в колонках)
    if Y.ndim == 1: Y = Y[:, np.newaxis]
    if X.ndim == 1: X = X[:, np.newaxis]
    
    num_samples, num_signals = Y.shape
    multiple_X = X.shape[1] > 1
    Y_out = np.zeros_like(Y, dtype=float)

    # Обертки для параметров, если они переданы как числа
    get_step = step_size if callable(step_size) else lambda x: step_size
    get_window = window_size if callable(window_size) else lambda x: window_size

    for ns in range(num_signals):
        curr_X = X[:, ns] if multiple_X else X[:, 0]
        curr_Y = Y[:, ns]
        
        # 1. Определение положений окон
        xp_list = []
        curr_p = max(0, curr_X[0])
        limit_X = curr_X[-1]
        
        while curr_p <= limit_X:
            xp_list.append(curr_p)
            curr_p += get_step(curr_p)
            if len(xp_list) > 1000: break # Предохранитель
            
        xp = np.array(xp_list)
        xw = np.array([get_window(p) for p in xp])
        mid_points = xp + xw / 2
        
        # 2. Оценка фона (Baseline points)
        we = np.zeros_like(xp)
        for nw in range(len(xp)):
            # Индексы точек, попадающих в окно [xp, xp + xw]
            mask = (curr_X >= xp[nw]) & (curr_X <= (xp[nw] + xw[nw]))
            subw = curr_Y[mask]
            
            if subw.size > 0:
                if estimation_method == 'quantile':
                    we[nw] = np.quantile(subw, quantile_value)
                else: # 'mean' или аналог
                    we[nw] = np.mean(subw)
            else:
                we[nw] = we[nw-1] if nw > 0 else np.min(curr_Y)

        # 3. Регрессия (Интерполяция базовой линии)
        if len(xp) > 1:
            if regression_method == 'pchip':
                interp_func = PchipInterpolator(mid_points, we, extrapolate=True)
                baseline = interp_func(curr_X)
            else:
                kind = 'linear' if regression_method == 'linear' else 'cubic'
                interp_func = interp1d(mid_points, we, kind=kind, fill_value="extrapolate")
                baseline = interp_func(curr_X)
        else:
            baseline = np.full_like(curr_X, we[0])

        # 4. Визуализация
        if show_plot == ns:
            _plot_results(curr_X, curr_Y, baseline, mid_points, we, ns)

        # 5. Применение коррекции
        if preserve_heights:
            # Формула из оригинального MATLAB кода: (Y - b) / (1 - b/max(Y))
            k = 1 - baseline / (np.max(curr_Y) + 1e-9)
            Y_out[:, ns] = (curr_Y - baseline) / (k + 1e-9)
        else:
            Y_out[:, ns] = curr_Y - baseline

    return Y_out.squeeze()

def _plot_results(x, y, baseline, pts_x, pts_y, idx):
    """Вспомогательная функция для отрисовки."""
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Original Signal', color='gray', alpha=0.5)
    plt.plot(x, baseline, 'r-', linewidth=2, label='Regressed Baseline')
    plt.scatter(pts_x, pts_y, color='black', marker='x', label='Estimated Points')
    plt.title(f"Baseline Correction - Signal {idx}")
    plt.legend()
    plt.grid(True)
   