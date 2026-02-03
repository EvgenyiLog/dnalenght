import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths
from pybaselines import Baseline
from typing import Tuple, Dict, Optional, Union, List
import numpy.typing as npt
from .subtract_reference_from_columns import subtract_reference_from_columns
from .msbackadj import msbackadj


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Блок подготовки данных для каждой колонки DataFrame:
    - транспонирование (строки -> колонки интенсивностей)
    - первые 50 значений для удаления шума  
    - baseline correction (pybaselines asls)
    - отрицательные к нулю
    
    Args:
        df: DataFrame где строки = точки спектра, колонки = спектры
    
    Returns:
        DataFrame с препроцессинговыми данными (те же размеры)
    """

    baseline_fitter = Baseline()
    processed_columns = []

    for col_name in df.columns:
        data = df[col_name].values.astype(np.float64)

        # Удаление шума (первые 50)
        data_trim = data[50:]

        # Baseline correction
        baseline, _ = baseline_fitter.asls(data_trim, lam=1e5, p=0.01)
        data_corrected = np.maximum(data_trim - baseline, 0)

        # Заполняем первые 50 нулями для сохранения формы
        result = np.zeros_like(data)
        result[50:] = data_corrected

        processed_columns.append(result)

    # Возвращаем в DataFrame той же формы что исходный
    result_df = pd.DataFrame(np.array(processed_columns),
                             index=df.index,
                             columns=df.columns)
    return result_df
