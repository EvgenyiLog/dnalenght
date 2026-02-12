import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Union
import warnings

def calculate_calibration_curve(
    df: pd.DataFrame,
    x_col: str = "Known_Size",
    y_col: str = "Peak_Position",
    method: str = "linear",
    log_x: bool = True,
    log_y: bool = False,
    selected_mask: Optional[pd.Series] = None,
    plot: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Калибровочная кривая",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None
) -> Dict[str, Union[float, np.ndarray, pd.Series, Dict]]:
    """
    Вычисляет калибровочную кривую для установления зависимости между 
    известными значениями (например, размером ДНК-фрагментов) и измеренными 
    позициями пиков в сигнале.
    
    Поддерживаемые методы калибровки:
    - 'linear': Y = a * X + b
    - 'log_linear': Y = a * log(X) + b (рекомендуется для ДНК-анализа)
    - 'poly2': Y = a * X² + b * X + c
    - 'power': Y = a * X^b
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными калибровки.
    x_col : str, optional (default="Known_Size")
        Имя колонки с известными значениями (например, размер маркера в п.н.).
    y_col : str, optional (default="Peak_Position")
        Имя колонки с измеренными позициями пиков (индексы в сигнале).
    method : str, optional (default="linear")
        Метод аппроксимации: 'linear', 'log_linear', 'poly2', 'power'.
    log_x : bool, optional (default=True)
        Применять логарифмическую трансформацию к X (только для 'linear').
    log_y : bool, optional (default=False)
        Применять логарифмическую трансформацию к Y (только для 'linear').
    selected_mask : pd.Series, optional (default=None)
        Булев маска для выбора только отмеченных точек калибровки.
        Если None — используются все строки датафрейма.
    plot : bool, optional (default=True)
        Строить график калибровочной кривой с аннотациями.
    figsize : tuple, optional (default=(10, 6))
        Размер фигуры для графика.
    title : str, optional (default="Калибровочная кривая")
        Заголовок графика.
    xlabel : str, optional (default=None)
        Подпись оси X. Если None — автоматическое определение.
    ylabel : str, optional (default=None)
        Подпись оси Y. Если None — автоматическое определение.
    
    Возвращает:
    ----------
    dict : Словарь с результатами:
        - 'model': dict с параметрами модели (коэффициенты, уравнение)
        - 'r_squared': float, коэффициент детерминации R²
        - 'residuals': np.ndarray, остатки модели
        - 'predicted_y': np.ndarray, предсказанные значения Y
        - 'x_transformed': np.ndarray, трансформированные X (если применялась лог-трансформация)
        - 'y_transformed': np.ndarray, трансформированные Y (если применялась лог-трансформация)
        - 'used_points': int, количество использованных точек калибровки
        - 'equation_str': str, строковое представление уравнения
    
    Пример использования:
    --------------------
    >>> # Данные калибровки для ДНК-маркера
    >>> df_cal = pd.DataFrame({
    ...     "Known_Size": [100, 200, 400, 600, 800, 1000, 1500, 2000],
    ...     "Peak_Position": [43, 87, 148, 187, 211, 235, 277, 309],
    ...     "Selected": ["✓", "✓", "", "✓", "✓", "✓", "✓", ""]
    ... })
    >>> 
    >>> # Вычисление калибровки только по отмеченным точкам
    >>> mask = df_cal["Selected"] == "✓"
    >>> result = calculate_calibration_curve(
    ...     df_cal, 
    ...     x_col="Known_Size", 
    ...     y_col="Peak_Position",
    ...     method="log_linear",
    ...     selected_mask=mask,
    ...     title="Калибровка ДНК-маркера (100-2000 п.н.)"
    ... )
    >>> 
    >>> print(f"Уравнение: {result['equation_str']}")
    >>> print(f"R² = {result['r_squared']:.4f}")
    >>> 
    >>> # Расчет неизвестного размера по позиции пика
    >>> unknown_position = 256
    >>> if result['model']['type'] == 'log_linear':
    ...     unknown_size = np.exp((unknown_position - result['model']['intercept']) / result['model']['slope'])
    ...     print(f"Размер фрагмента на позиции {unknown_position}: {unknown_size:.0f} п.н.")
    """
    
    # === 1. Валидация входных данных ===
    if x_col not in df.columns:
        raise ValueError(f"Колонка '{x_col}' не найдена в датафрейме. Доступные: {list(df.columns)}")
    if y_col not in df.columns:
        raise ValueError(f"Колонка '{y_col}' не найдена в датафрейме. Доступные: {list(df.columns)}")
    
    if selected_mask is not None:
        if len(selected_mask) != len(df):
            raise ValueError("Длина маски selected_mask должна совпадать с длиной датафрейма")
        df_used = df[selected_mask].copy()
        if len(df_used) < 2:
            raise ValueError(f"Недостаточно отмеченных точек для калибровки: {len(df_used)} (минимум 2)")
    else:
        df_used = df.copy()
        if len(df_used) < 2:
            raise ValueError(f"Недостаточно точек в датафрейме: {len(df_used)} (минимум 2)")
    
    # === 2. Подготовка данных ===
    x_raw = df_used[x_col].astype(float).values
    y_raw = df_used[y_col].astype(float).values
    
    # Проверка на положительные значения для логарифмов
    if (method in ["log_linear", "power"] or log_x) and np.any(x_raw <= 0):
        raise ValueError("Для логарифмической трансформации все значения X должны быть > 0")
    if log_y and np.any(y_raw <= 0):
        raise ValueError("Для логарифмической трансформации все значения Y должны быть > 0")
    
    # === 3. Выбор метода аппроксимации ===
    model_info = {"type": method}
    residuals = None
    predicted_y = None
    equation_str = ""
    
    if method == "log_linear":
        # Y = a * log(X) + b  →  линейная регрессия в координатах (log(X), Y)
        x_transformed = np.log(x_raw)
        y_transformed = y_raw
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_transformed, y_transformed)
        predicted_y_transformed = slope * x_transformed + intercept
        predicted_y = predicted_y_transformed
        residuals = y_transformed - predicted_y_transformed
        
        model_info.update({
            "slope": float(slope),
            "intercept": float(intercept),
            "std_err": float(std_err)
        })
        r_squared = r_value ** 2
        equation_str = f"Y = {slope:.4f} * ln(X) + {intercept:.4f}"
        
    elif method == "linear":
        # Y = a * X + b (с опциональной лог-трансформацией)
        x_transformed = np.log(x_raw) if log_x else x_raw
        y_transformed = np.log(y_raw) if log_y else y_raw
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_transformed, y_transformed)
        predicted_y_transformed = slope * x_transformed + intercept
        predicted_y = np.exp(predicted_y_transformed) if log_y else predicted_y_transformed
        residuals = y_transformed - predicted_y_transformed
        
        model_info.update({
            "slope": float(slope),
            "intercept": float(intercept),
            "std_err": float(std_err),
            "log_x": log_x,
            "log_y": log_y
        })
        r_squared = r_value ** 2
        
        if log_x and log_y:
            equation_str = f"ln(Y) = {slope:.4f} * ln(X) + {intercept:.4f}"
        elif log_x:
            equation_str = f"Y = {slope:.4f} * ln(X) + {intercept:.4f}"
        elif log_y:
            equation_str = f"ln(Y) = {slope:.4f} * X + {intercept:.4f}"
        else:
            equation_str = f"Y = {slope:.4f} * X + {intercept:.4f}"
    
    elif method == "poly2":
        # Y = a * X² + b * X + c
        x_transformed = x_raw
        y_transformed = y_raw
        
        coeffs = np.polyfit(x_transformed, y_transformed, 2)
        predicted_y = np.polyval(coeffs, x_transformed)
        residuals = y_transformed - predicted_y
        
        model_info.update({
            "coeffs": coeffs.tolist(),  # [a, b, c] для a*x² + b*x + c
            "a": float(coeffs[0]),
            "b": float(coeffs[1]),
            "c": float(coeffs[2])
        })
        
        # Расчет R² вручную для полинома
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_transformed - np.mean(y_transformed)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        equation_str = f"Y = {coeffs[0]:.6f} * X² + {coeffs[1]:.4f} * X + {coeffs[2]:.4f}"
    
    elif method == "power":
        # Y = a * X^b  →  ln(Y) = ln(a) + b * ln(X)
        x_transformed = np.log(x_raw)
        y_transformed = np.log(y_raw)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_transformed, y_transformed)
        predicted_y_transformed = slope * x_transformed + intercept
        predicted_y = np.exp(predicted_y_transformed)
        residuals = y_transformed - predicted_y_transformed
        
        model_info.update({
            "a": float(np.exp(intercept)),  # a = exp(intercept)
            "b": float(slope),
            "std_err": float(std_err)
        })
        r_squared = r_value ** 2
        equation_str = f"Y = {np.exp(intercept):.4f} * X^{slope:.4f}"
    
    else:
        raise ValueError(f"Неизвестный метод '{method}'. Допустимые: 'linear', 'log_linear', 'poly2', 'power'")
    
    # === 4. Визуализация ===
    if plot:
        plt.figure(figsize=figsize)
        
        # Определение подписей осей
        if xlabel is None:
            xlabel = f"Известный размер ({x_col})" if not (method == "log_linear" or (method == "linear" and log_x)) else f"ln(Известный размер)"
        if ylabel is None:
            ylabel = f"Позиция пика ({y_col})" if not (method == "power" or (method == "linear" and log_y)) else f"ln(Позиция пика)"
        
        # Точки калибровки
        plt.scatter(
            x_transformed if method in ["log_linear", "power"] or (method == "linear" and log_x) else x_raw,
            y_transformed if method in ["power"] or (method == "linear" and log_y) else y_raw,
            color='red', s=100, zorder=5, label='Точки калибровки', edgecolors='black', linewidth=1.5
        )
        
        # Аннотации точек
        for i, (x, y) in enumerate(zip(
            x_transformed if method in ["log_linear", "power"] or (method == "linear" and log_x) else x_raw,
            y_transformed if method in ["power"] or (method == "linear" and log_y) else y_raw
        )):
            label = f"{x_raw[i]:.0f}" if method in ["log_linear", "power"] or log_x else f"{x:.0f}"
            plt.annotate(
                label,
                (x, y),
                xytext=(8, 8),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
               
            )
        
        # Калибровочная кривая
        x_fit = np.linspace(x_transformed.min(), x_transformed.max(), 200) if method in ["log_linear", "power"] or (method == "linear" and log_x) else np.linspace(x_raw.min(), x_raw.max(), 200)
        
        if method == "poly2":
            y_fit = np.polyval(coeffs, x_fit)
        elif method in ["log_linear", "linear", "power"]:
            y_fit = slope * x_fit + intercept
        
        plt.plot(x_fit, y_fit, 'b-', linewidth=2, label=f'Калибровка (R² = {r_squared:.4f})')
        
        # Настройки графика
        plt.xlabel(xlabel, fontsize=12, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Добавляем уравнение на график
        plt.text(
            0.05, 0.95, 
            f"Уравнение:\n{equation_str}\nR² = {r_squared:.4f}\nТочек: {len(df_used)}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        plt.tight_layout()
       
    
    # === 5. Формирование результата ===
    result = {
        "model": model_info,
        "r_squared": float(r_squared),
        "residuals": residuals,
        "predicted_y": predicted_y,
        "x_transformed": x_transformed,
        "y_transformed": y_transformed if method in ["power"] or (method == "linear" and log_y) else y_raw,
        "used_points": len(df_used),
        "equation_str": equation_str,
        "raw_x": x_raw,
        "raw_y": y_raw
    }
    
    return result


# === Вспомогательная функция: расчет неизвестного размера по калибровке ===
def estimate_unknown_size(
    calibration_result: Dict,
    peak_position: float,
    method: Optional[str] = None
) -> float:
    """
    Рассчитывает неизвестный размер фрагмента по его позиции пика 
    используя предварительно вычисленную калибровочную кривую.
    
    Параметры:
    ----------
    calibration_result : dict
        Результат функции calculate_calibration_curve.
    peak_position : float
        Позиция неизвестного пика в сигнале.
    method : str, optional
        Метод калибровки (берется из calibration_result если не указан).
    
    Возвращает:
    ----------
    float : Оцененный размер фрагмента.
    """
    model = calibration_result["model"]
    method = method or model["type"]
    
    if method == "log_linear":
        # Y = a * ln(X) + b  →  X = exp((Y - b) / a)
        return np.exp((peak_position - model["intercept"]) / model["slope"])
    
    elif method == "linear":
        if model.get("log_x") and not model.get("log_y"):
            # Y = a * ln(X) + b  →  X = exp((Y - b) / a)
            return np.exp((peak_position - model["intercept"]) / model["slope"])
        elif not model.get("log_x") and not model.get("log_y"):
            # Y = a * X + b  →  X = (Y - b) / a
            return (peak_position - model["intercept"]) / model["slope"]
        elif model.get("log_x") and model.get("log_y"):
            # ln(Y) = a * ln(X) + b  →  X = exp((ln(Y) - b) / a)
            return np.exp((np.log(peak_position) - model["intercept"]) / model["slope"])
        else:
            raise NotImplementedError("Комбинация трансформаций не поддерживается")
    
    elif method == "poly2":
        # Решаем квадратное уравнение: a * X² + b * X + c - Y = 0
        a, b, c = model["a"], model["b"], model["c"]
        # a * X² + b * X + (c - Y) = 0
        discriminant = b**2 - 4 * a * (c - peak_position)
        if discriminant < 0:
            raise ValueError("Нет действительных корней для заданной позиции пика")
        # Выбираем положительный корень
        x1 = (-b + np.sqrt(discriminant)) / (2 * a)
        x2 = (-b - np.sqrt(discriminant)) / (2 * a)
        return max(x1, x2)  # Предполагаем, что размер должен быть положительным
    
    elif method == "power":
        # Y = a * X^b  →  X = (Y / a)^(1/b)
        return (peak_position / model["a"]) ** (1 / model["b"])
    
    else:
        raise ValueError(f"Неизвестный метод калибровки: {method}")


# === Пример использования ===
if __name__ == "__main__":
    # Создаем пример данных для ДНК-маркера
    df_calibration = pd.DataFrame({
        "Known_Size_bp": [100, 200, 400, 600, 800, 1000, 1500, 2000, 3000],
        "Peak_Index": [43, 87, 148, 187, 211, 235, 277, 309, 362],
        "Selected": ["✓", "✓", "", "✓", "✓", "✓", "✓", "✓", ""]
    })
    
    # Выполняем калибровку только по отмеченным точкам
    mask = df_calibration["Selected"] == "✓"
    
    result = calculate_calibration_curve(
        df_calibration,
        x_col="Known_Size_bp",
        y_col="Peak_Index",
        method="log_linear",  # Оптимально для ДНК
        selected_mask=mask,
        title="Калибровочная кривая ДНК-маркера",
        xlabel="ln(Размер фрагмента, п.н.)",
        ylabel="Позиция пика (индекс)"
    )
    
    print("=" * 60)
    print("РЕЗУЛЬТАТЫ КАЛИБРОВКИ")
    print("=" * 60)
    print(f"Метод: {result['model']['type']}")
    print(f"Уравнение: {result['equation_str']}")
    print(f"Коэффициент детерминации R²: {result['r_squared']:.6f}")
    print(f"Использовано точек: {result['used_points']}")
    print(f"Остаточная стандартная ошибка: {np.std(result['residuals']):.4f}")
    print("=" * 60)
    
    # Оцениваем размер неизвестного фрагмента
    unknown_peak = 256
    estimated_size = estimate_unknown_size(result, unknown_peak)
    print(f"\nНеизвестный пик на позиции {unknown_peak} → размер ≈ {estimated_size:.0f} п.н.")
    
    # Дополнительные неизвестные пики
    test_peaks = [256, 332, 348, 384, 406]
    print("\nОценка размеров неизвестных фрагментов:")
    print("-" * 40)
    for peak in test_peaks:
        size = estimate_unknown_size(result, peak)
        print(f"  Пик {peak:3d} → {size:7.0f} п.н.")