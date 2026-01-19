import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths
from pybaselines import Baseline
from typing import Tuple, Dict, Optional, Union, List
import numpy.typing as npt
from analyze_single_spectrum import analyze_single_spectrum

def analyze_peaks(df_processed: pd.DataFrame,
                  x: Optional[npt.NDArray[np.float64]] = None,
                  first_reper_idx: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Анализ пиков для каждой колонки препроцессингованного DataFrame.
    
    Args:
        df_processed: DataFrame после preprocess_data()
        x: опционально, ось m/z или индексы
        first_reper_idx: индекс первого репера стандарта длин
    
    Returns:
        - selected_peaks_df: DataFrame с индексами выбранных пиков (колонки=спектры)
        - scores_df: DataFrame с баллами пиков
        - properties: словарь с all_peaks, prominences, widths, heights для всех спектров
    """
    if x is None:
        x = np.arange(len(df_processed))
    
    all_selected = []
    all_scores = []
    all_properties = {'all_peaks': [], 'prominences': [], 'widths': [], 'heights': []}
    
    for col_name in df_processed.columns:
        data_processed = df_processed[col_name].values
        
        # Анализ пиков для одного спектра
        selected, scores, props = analyze_single_spectrum(data_processed, x, first_reper_idx)
        
        all_selected.append(selected)
        all_scores.append(scores)
        all_properties['all_peaks'].append(props.get('all_peaks', np.array([])))
        all_properties['prominences'].append(props.get('prominences', np.array([])))
        all_properties['widths'].append(props.get('widths', np.array([])))
        all_properties['heights'].append(props.get('heights', np.array([])))
    
    # DataFrame с результатами
    selected_df = pd.DataFrame(np.array(all_selected).T, 
                              index=[f'peak_{i}' for i in range(len(all_selected[0]))] 
                              if len(all_selected) > 0 and len(all_selected[0]) > 0 else [],
                              columns=df_processed.columns)
    
    scores_df = pd.DataFrame(np.array(all_scores).T, columns=df_processed.columns)
    
    return selected_df, scores_df, all_properties