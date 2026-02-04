import numpy as np
import pandas as pd
from typing import Union, Literal, Optional, Tuple, List, Dict
from pybaselines import Baseline


def correct_baseline(
    data: Union[pd.Series, np.ndarray, pd.DataFrame],
    column: Optional[str] = None,
    method: Literal[
        'iarpls', 'aspls', 'modpoly', 'psalsa', 'airpls', 'iasls',
        'beads', 'dietrich', 'rbf', 'corner_clip'
    ] = 'iarpls',
    lam: Optional[float] = None,
    x_data: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Коррекция базовой линии сигнала с использованием методов из pybaselines.
    
    Parameters
    ----------
    data : pd.Series | np.ndarray | pd.DataFrame
        Входной сигнал или DataFrame с колонкой сигнала.
    column : str, optional
        Имя колонки в DataFrame. Обязательно, если data — DataFrame.
    method : str, default='iarpls'
        Метод коррекции базовой линии:
        - 'iarpls'     : итеративный асимметричный взвешенный штрафной МНК (рекомендуется для большинства случаев)
        - 'aspls'      : асимметричный штрафной МНК с адаптивными весами (устойчив к шуму)
        - 'airpls'     : улучшенный итеративный асимметричный МНК
        - 'iasls'      : итеративный асимметричный сплайн
        - 'modpoly'    : модифицированная полиномиальная аппроксимация (для гладкого фона)
        - 'psalsa'     : штрафной сплайн с асимметричными весами
        - 'beads'      : BEADS — разложение на компоненты с частотной фильтрацией (отлично для периодических фонов)
        - 'dietrich'   : алгоритм Дитриха (для спектроскопических данных)
        - 'rbf'        : радиальные базисные функции (гибкая аппроксимация)
        - 'corner_clip': метод отсечения углов (быстрый, для простых фонов)
    lam : float, optional
        Параметр сглаживания (штраф за кривизну). Автоматически подбирается для некоторых методов:
        - iarpls/aspls/airpls/iasls/psalsa: 1e4–1e8 (по умолчанию 1e6)
        - beads: не используется (регулируется через freq_cutoff, lam_0/1/2)
        - modpoly: не используется (регулируется через poly_order)
        - dietrich/rbf/corner_clip: специфичные параметры
    x_data : np.ndarray, optional
        Ось X (время, длина волны и т.д.). Если не задано — используется линейная шкала индексов.
    **kwargs : dict
        Дополнительные параметры для выбранного метода:
        - Для iarpls/aspls/airpls: ratio (0.001–0.1) — асимметрия весов
        - Для modpoly: poly_order (1–3) — порядок полинома
        - Для psalsa: k (0.01–0.1) — коэффициент асимметрии
        - Для beads: 
            * freq_cutoff (0.001–0.01) — порог частоты для разделения сигнала/фона
            * lam_0, lam_1, lam_2 — параметры штрафа (по умолчанию 3, 0.05, 0.2)
            * asymmetry (1–5) — степень асимметрии
        - Для dietrich: poly_order, smooth_half_window
        - Для rbf: width, num_knots
        - Для corner_clip: max_iterations
    
    Returns
    -------
    corrected : np.ndarray
        Сигнал после вычитания базовой линии.
    baseline : np.ndarray
        Оценённая базовая линия.
    
    Examples
    --------
    >>> # Рекомендуемый метод для большинства сигналов
    >>> corrected, baseline = correct_baseline(df['signal'], method='iarpls', lam=1e7, ratio=0.01)
    
    >>> # BEADS для сигналов с периодическим фоном (например, дыхание в ЭКГ)
    >>> corrected, baseline = correct_baseline(
    ...     df['signal'], 
    ...     method='beads', 
    ...     freq_cutoff=0.002, 
    ...     asymmetry=3
    ... )
    
    >>> # Использование реальной оси X (например, длина волны в нм)
    >>> x = np.linspace(400, 800, len(df))
    >>> corrected, baseline = correct_baseline(df['absorbance'], x_data=x, method='aspls', lam=1e5)
    
    ⚠️ ВАЖНО: Не применяйте несколько методов последовательно (например, iarpls → aspls) — 
    это приведёт к искажению сигнала. Выбирайте ОДИН наиболее подходящий метод.
    """
    # === Извлечение данных ===
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("Параметр 'column' обязателен при передаче DataFrame")
        y = data[column].values.astype(float)
    elif isinstance(data, pd.Series):
        y = data.values.astype(float)
    elif isinstance(data, np.ndarray):
        y = data.astype(float)
    else:
        raise TypeError("data должен быть pd.Series, np.ndarray или pd.DataFrame")

    # === Валидация сигнала ===
    if y.ndim != 1:
        raise ValueError(f"Ожидался одномерный сигнал, получено измерение: {y.ndim}")
    if len(y) < 10:
        raise ValueError("Сигнал слишком короткий для коррекции базовой линии (< 10 точек)")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Сигнал содержит NaN или бесконечные значения")

    # === Подготовка оси X ===
    if x_data is not None:
        if len(x_data) != len(y):
            raise ValueError("Длина x_data должна совпадать с длиной сигнала")
        x = np.asarray(x_data, dtype=float)
    else:
        x = np.arange(len(y), dtype=float)

    # === Инициализация корректора ===
    baseline_fitter = Baseline(x_data=x, check_finite=False)

    # === Автоматический подбор lam для методов, где он критичен ===
    if lam is None:
        method_lower = method.lower()
        if method_lower in ('iarpls', 'aspls', 'airpls', 'iasls', 'psalsa'):
            lam = 1e6
        elif method_lower == 'modpoly':
            lam = None  # не используется
        else:
            lam = 1e5  # значение по умолчанию для остальных

    # === Применение выбранного метода ===
    method_lower = method.lower()
    try:
        if method_lower == 'iarpls':
            baseline, _ = baseline_fitter.iarpls(y, lam=lam, **kwargs)
        elif method_lower == 'aspls':
            baseline, _ = baseline_fitter.aspls(y, lam=lam, **kwargs)
        elif method_lower == 'modpoly':
            poly_order = kwargs.pop('poly_order', 2)
            baseline, _ = baseline_fitter.modpoly(y, poly_order=poly_order, **kwargs)
        elif method_lower == 'psalsa':
            baseline, _ = baseline_fitter.psalsa(y, lam=lam, **kwargs)
        elif method_lower == 'airpls':
            baseline, _ = baseline_fitter.airpls(y, lam=lam, **kwargs)
        elif method_lower == 'iasls':
            baseline, _ = baseline_fitter.iasls(y, lam=lam, **kwargs)
        elif method_lower == 'beads':
            # BEADS не использует lam — передаём только специфичные параметры
            beads_params = {
                'freq_cutoff': kwargs.get('freq_cutoff', 0.002),
                'lam_0': kwargs.get('lam_0', 3),
                'lam_1': kwargs.get('lam_1', 0.05),
                'lam_2': kwargs.get('lam_2', 0.2),
                'asymmetry': kwargs.get('asymmetry', 3)
            }
            baseline, _ = baseline_fitter.beads(y, **beads_params)
        elif method_lower == 'dietrich':
            baseline, _ = baseline_fitter.dietrich(y, **kwargs)
        elif method_lower == 'rbf':
            baseline, _ = baseline_fitter.rbf(y, **kwargs)
        elif method_lower == 'corner_clip':
            baseline, _ = baseline_fitter.corner_clip(y, **kwargs)
        else:
            raise ValueError(
                f"Неизвестный метод '{method}'. Доступные: iarpls, aspls, modpoly, psalsa, "
                f"airpls, iasls, beads, dietrich, rbf, corner_clip"
            )
    except Exception as e:
        raise RuntimeError(
            f"Ошибка при применении метода '{method}': {e}\n"
            f"Подсказка: проверьте параметры метода в документации pybaselines"
        ) from e

    # === Коррекция сигнала ===
    corrected = y - baseline

    return corrected, baseline


# ============================================================================
# Дополнительная утилита: сравнение методов (для подбора оптимального)
# ============================================================================
def compare_baseline_methods(
    signal: np.ndarray,
    methods: Optional[List[str]] = None,
    lam: float = 1e6,
    x_data: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Сравнение нескольких методов коррекции базовой линии на одном сигнале.
    
    Возвращает словарь: {метод: (скорректированный_сигнал, базовая_линия)}
    
    Пример:
        results = compare_baseline_methods(
            df['signal'].values,
            methods=['iarpls', 'beads', 'aspls'],
            lam=1e7,
            freq_cutoff=0.002
        )
    """
    if methods is None:
        methods = ['iarpls', 'aspls', 'beads']
    
    results = {}
    for method in methods:
        try:
            corrected, baseline = correct_baseline(
                signal,
                method=method,
                lam=lam if method not in ('beads', 'modpoly') else None,
                x_data=x_data,
                **kwargs
            )
            results[method] = (corrected, baseline)
        except Exception as e:
            results[method] = (None, f"Ошибка: {e}")
    
    return results