import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths
from pybaselines import Baseline
from typing import Tuple, Dict, Optional, Union, List
import numpy.typing as npt
from .select_top_peaks import select_top_peaks
from .score_peaks import score_peaks


def analyze_single_spectrum(data: npt.NDArray[np.float64],
                            x: npt.NDArray[np.float64],
                            first_reper_idx: Optional[int]) -> Tuple[npt.NDArray[np.intp],
                                                                     npt.NDArray[np.float64],
                                                                     Dict]:
    """Анализ одного спектра (извлечена для переиспользования)."""
    # SG фильтр
    data_sg = savgol_filter(data, window_length=11, polyorder=3)

    # Производные
    der1 = np.gradient(data_sg)
    der2 = np.gradient(der1)
    der2_flipped = np.maximum(-der2, 0)

    # findpeaks
    peaks, props = find_peaks(der2_flipped, prominence=0.1)
    if len(peaks) == 0:
        return np.array([]), np.array([]), {}

    heights = props['peak_heights']
    prominences = props['prominences']
    widths = peak_widths(der2_flipped, peaks, rel_height=0.5)[0]

    # Фильтр индексов
    valid_mask = (peaks >= 15) & (peaks < len(der2_flipped) - 10)
    peaks = peaks[valid_mask]
    heights, prominences, widths = [p[valid_mask]
                                    for p in (heights, prominences, widths)]

    if len(peaks) == 0:
        return np.array([]), np.array([]), {}

    # Скоринг
    scores = score_peaks(heights, prominences, widths,
                         der2_flipped, peaks, first_reper_idx)
    selected_peaks = select_top_peaks(peaks, scores)

    return selected_peaks, scores, {
        'all_peaks': peaks, 'prominences': prominences,
        'widths': widths, 'heights': heights
    }
