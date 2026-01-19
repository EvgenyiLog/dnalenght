import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths
from pybaselines import Baseline
from typing import Tuple, Dict, Optional, Union, List
import numpy.typing as npt


def select_top_peaks(peaks: npt.NDArray[np.intp], scores: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    scores_round = np.round(scores)
    max_score = np.max(scores_round)
    top_mask = scores_round == max_score
    top_peaks = peaks[top_mask]
    
    if len(top_peaks) >= 2:
        return np.sort(top_peaks)[[0, -1]]  # первый и последний
    return top_peaks