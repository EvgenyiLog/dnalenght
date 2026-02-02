from typing import Literal
from numpy.typing import NDArray
import pywt
import numpy as np


def wden(
    signal: NDArray,
    tptr: Literal['sqtwolog'] = 'sqtwolog',
    sorh: Literal['s', 'h'] = 's',
    scal: Literal['sln'] = 'sln',
    level: int = 1,
    wname: str = 'sym2'
) -> NDArray[np.floating]:
    """
    Partial reproduction of MATLAB's `wden` function for denoising a signal using wavelet transform.
    This version implements only a subset of the original functionality.

    Parameters:
    - signal: 1D array-like, input signal to denoise
    - tptr: threshold selection rule (only 'sqtwolog' is supported)
    - sorh: thresholding type, 's' for soft or 'h' for hard
    - scal: threshold rescaling method (only 'sln' is supported: single level noise estimation)
    - level: decomposition level for wavelet transform
    - wname: wavelet name (e.g., 'sym2')

    Returns:
    - Denoised signal as a NumPy array, same length as input
    """
    N = len(signal)

    # Perform wavelet decomposition
    cA, cD = pywt.wavedec(signal, wname, level=level)

    # Estimate noise using median absolute deviation of the detail coefficients
    sigma = np.median(np.abs(cD)) / 0.6745

    # Compute universal threshold (sqtwolog rule)
    threshold = sigma * np.sqrt(2 * np.log(2 * len(cD)))

    # Apply thresholding to the detail coefficients
    cD_thresh = pywt.threshold(cD, threshold, mode='hard' if sorh == 'h' else 'soft')

    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec([cA, cD_thresh], wname)[:N]  # Truncate in case output is longer due to padding
