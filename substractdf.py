import pandas as pd
from typing import Union


def subtract_column_min(df: pd.DataFrame) -> pd.DataFrame:
    """Вычитает минимальное значение каждой колонки из всех значений этой колонки.

    Эта функция нормализует датафрейм путём центрирования данных относительно минимума.

    После применения, минимальное значение в каждой колонке становится равным нулю,
    а все остальные значения — неотрицательными (если исходные данные были неотрицательными).

    Args:
        df (pd.DataFrame): Входной датафрейм с числовыми колонками.

    Returns:
        pd.DataFrame: Новый датафрейм, где из каждой колонки вычтен её минимум.

    Raises:
        ValueError: Если датафрейм пуст или содержит нечисловые колонки.

    Example:
        >>> df = pd.DataFrame({'A': [2, 5, 3], 'B': [10, 15, 12]})
        >>> subtract_column_min(df)
           A  B
        0  0  0
        1  3  5
        2  1  2

    Note:
        Функция не изменяет исходный датафрейм, а возвращает новый.
        Работает корректно только с числовыми данными.
    """
    if df.empty:
        raise ValueError("Датафрейм пуст.")

    # Проверяем, что все колонки числовые
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
        raise ValueError("Все колонки должны быть числовыми.")

    return df - df.min(numeric_only=True)


def subtract_percentile_norm(df: pd.DataFrame, q: float = 1.0) -> pd.DataFrame:
    """Вычитает заданный перцентиль по каждой колонке."""
    return df - df.quantile(q / 100)