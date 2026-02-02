import numpy as np
import pandas as pd
from typing import Union, Literal, Optional, Tuple
from pybaselines import Baseline


def correct_baseline(
    data: Union[pd.Series, np.ndarray, pd.DataFrame],
    column: Optional[str] = None,
    method: Literal['iarpls', 'aspls', 'modpoly', 'psalsa', 'airpls', 'iasls'] = 'iarpls',
    lam: float = 1e6,
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
        - 'iarpls': итеративный асимметричный взвешенный штрафной МНК (рекомендуется)
        - 'aspls': асимметричный штрафной МНК с адаптивными весами
        - 'modpoly': модифицированная полиномиальная аппроксимация
        - 'psalsa': штрафной сплайн с асимметричными весами
        - 'airpls': улучшенный итеративный асимметричный МНК
        - 'iasls': итеративный асимметричный сплайн
    lam : float, default=1e6
        Параметр сглаживания (штраф за кривизну). Чем больше — тем плавнее базовая линия.
        Типичные значения: 1e4–1e8 (зависит от разрешения сигнала).
    **kwargs : dict
        Дополнительные параметры для выбранного метода:
        - ratio : float (для iarpls/aspls/airpls) — асимметрия весов, типично 0.001–0.1
        - poly_order : int (для modpoly) — порядок полинома, типично 1–3
        - k : float (для psalsa) — коэффициент асимметрии, типично 0.01–0.1
    
    Returns
    -------
    corrected : np.ndarray
        Сигнал после вычитания базовой линии.
    baseline : np.ndarray
        Оценённая базовая линия.
    
    Examples
    --------
    >>> # Для Series
    >>> corrected, baseline = correct_baseline(df['dR110'], method='iarpls', lam=1e7, ratio=0.01)
    >>> df['dR110_corr'] = corrected
    
    >>> # Для DataFrame с указанием колонки
    >>> corrected, baseline = correct_baseline(df, column='dR110', method='aspls', lam=1e5)
    
    >>> # Быстрая коррекция с сохранением в новый столбец
    >>> df['dR110_corr'], df['baseline'] = correct_baseline(df['dR110'])
    """
    # Извлечение массива значений сигнала
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
    
    # Валидация сигнала
    if y.ndim != 1:
        raise ValueError(f"Ожидался одномерный сигнал, получено измерение: {y.ndim}")
    if len(y) < 10:
        raise ValueError("Сигнал слишком короткий для коррекции базовой линии (< 10 точек)")
    
    # Ось x (линейная шкала индексов)
    x = np.arange(len(y), dtype=float)
    
    # Инициализация корректора
    baseline_fitter = Baseline(x_data=x)
    
    # Выбор и применение метода
    method_lower = method.lower()
    if method_lower == 'iarpls':
        baseline, _ = baseline_fitter.iarpls(y, lam=lam, **kwargs)
    elif method_lower == 'aspls':
        baseline, _ = baseline_fitter.aspls(y, lam=lam, **kwargs)
    elif method_lower == 'modpoly':
        baseline, _ = baseline_fitter.modpoly(y, poly_order=kwargs.get('poly_order', 2), **kwargs)
    elif method_lower == 'psalsa':
        baseline, _ = baseline_fitter.psalsa(y, lam=lam, **kwargs)
    elif method_lower == 'airpls':
        baseline, _ = baseline_fitter.airpls(y, lam=lam, **kwargs)
    elif method_lower == 'iasls':
        baseline, _ = baseline_fitter.iasls(y, lam=lam, **kwargs)
    else:
        raise ValueError(
            f"Неизвестный метод '{method}'. Доступные: iarpls, aspls, modpoly, psalsa, airpls, iasls"
        )
    
    # Коррекция сигнала
    corrected = y - baseline
    
    return corrected, baseline


# Пример использования:
# corrected, baseline = correct_baseline(df['dR110'], method='iarpls', lam=1e7, ratio=0.01)
# df['dR110_corr'] = corrected
# df['baseline'] = baseline