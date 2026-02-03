import numpy as np
import pandas as pd
from pybaselines import Baseline
from scipy.signal import savgol_filter
# from pyyawt import wden  # pip install PyWavelets или pyyawt
from typing import Tuple, Dict, Optional, Union
import numpy.typing as npt
from .subtract_reference_from_columns import subtract_reference_from_columns
from .msbackadj import msbackadj


def process_frf_files(keyword_files: list[str],
                      other_files: list[str],
                      input_path: str) -> pd.DataFrame:
    """
    Полная обработка FRF файлов: вычитание референса, baseline correction, вейвлет-денойзинг.
    
    Args:
        keyword_files: список файлов ключевых каналов (реперы)
        other_files: список остальных файлов  
        input_path: путь к папке с файлами
    
    Returns:
        DataFrame с обработанными сигналами (исходные + corrected колонки)
    """
    # Загружаем channels_df (предполагается функция categorize_frf_files возвращает DataFrame)
    channels_df = load_frf_channels(keyword_files, other_files, input_path)

    # Вычитание референса на 50 единицах (как в примере)
    df = subtract_reference_from_columns(channels_df, 50)

    # Временная ось
    time = np.arange(len(channels_df))

    # Обработка каждой колонки с keyword_files
    baseline_fitter = Baseline()

    for col_name in keyword_files:
        if col_name in df.columns:
            signal = df[col_name].values

            print(f"Обрабатываем {col_name}, длина: {len(signal)}")

            # 1. msbackadj -> pybaselines asls
            baseline, _ = baseline_fitter.asls(signal, lam=1e5, p=0.01)
            signal_corrected = np.maximum(signal - baseline, 0)

            # 2. Вейвлет-денойзинг (pyyawt.wden эквивалент)
            # signal_denoised, CXD, LXD = wden(
            #     signal_corrected, 'sqtwolog', 's', 'sln', 1, 'sym2')

            # print(f"Длина после денойзинга: {len(signal_denoised)}")

            # # Добавляем обработанную колонку
            # df[f'{col_name}_corr'] = signal_denoised

    return df


def load_frf_channels(keyword_files: list[str],
                      other_files: list[str], 
                      input_path: str) -> pd.DataFrame:
    """
    Загрузка FRF каналов из файлов (заглушка - адаптируйте под ваш формат).
    
    Args:
        keyword_files: ключевые файлы
        other_files: остальные файлы
        input_path: путь
    
    Returns:
        channels_df: DataFrame со всеми каналами
    """
    # TODO: реализуйте загрузку ваших FRF файлов
    # Пример: pd.read_csv, или специальный парсер FRF
    all_files = keyword_files + other_files
    # Заглушка - замените на реальную загрузку
    channels_df = pd.DataFrame(np.random.rand(1000, len(all_files)), 
                               columns=all_files)
    return channels_df
