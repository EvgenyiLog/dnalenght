import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths
from pybaselines import Baseline
from typing import Tuple, Dict, Optional, Union, List
import numpy.typing as npt

# Вспомогательные функции score_peaks и select_top_peaks остаются без изменений
def score_peaks(heights: npt.NDArray[np.float64], 
                prominences: npt.NDArray[np.float64], 
                widths: npt.NDArray[np.float64],
                signal: npt.NDArray[np.float64],
                peaks: npt.NDArray[np.intp],
                first_reper_idx: Optional[int] = None) -> npt.NDArray[np.float64]:
    # ... (тот же код как раньше)
    scores = np.zeros(len(peaks))
    
    prom_log = np.round(np.log10(np.maximum(prominences, 1e-10)))
    max_order = np.max(prom_log)
    high_prom_mask = (prom_log == max_order)
    scores[high_prom_mask] += 1
    
    if len(widths) > 1:
        scores += np.argsort(-widths) / (len(widths) - 1)
    
    ratios = heights / np.maximum(widths, 1e-6)
    if len(ratios) > 1:
        scores += np.argsort(-ratios) / (len(ratios) - 1)
    
    height_thresh = np.mean(heights) / 3
    scores[heights > height_thresh] += 1
    
    for i, idx in enumerate(peaks):
        left, right = max(0, idx-4), min(len(signal), idx+4)
        local_max = np.argmax(signal[left:right]) + left
        h_l = signal[max(0, local_max-4)]
        h_r = signal[min(len(signal), local_max+4)]
        if h_l < signal[local_max] and h_r < signal[local_max]:
            scores[i] += 1
    
    return scores