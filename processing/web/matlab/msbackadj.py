import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator


def msbackadj(
    x: NDArray,
    y: NDArray,
    window_size: float = 140,
    step_size: float = 40,
    quantile_value: float = 0.1
) -> NDArray:
    """
    Approximate MATLAB's `msbackadj` function: subtracts a slowly varying baseline
    from a signal using a sliding window and quantile-based estimation.

    Parameters:
    - x: 1D array of x-values (must be monotonic)
    - y: 1D array of signal values (same length as x)
    - window_size: number of points in each sliding window (e.g. 140)
    - step_size: step size between consecutive windows (e.g. 40)
    - quantile_value: quantile to use for estimating the baseline in each window (e.g. 0.1 for 10%)

    Returns:
    - adjusted_signal: signal with estimated baseline subtracted
    """
    x = np.asarray(x)
    y = np.asarray(y)
    N = len(x)
    if N == 0:
        return np.empty(0)  # Return empty array for empty input

    baseline_x = []  # x-positions of baseline anchor points
    baseline_y = []  # corresponding y-values (quantiles)

    # Slide a window along the signal and compute quantile baseline at each step
    x_start = x[0]
    x_end = x[-1]

    while x_start <= x_end:
        # Window from x_start to x_start + window_size
        x_win_lo = x_start
        x_win_hi = x_start + window_size

        # Mask for points within the current window
        mask = (x >= x_win_lo) & (x <= x_win_hi)
        y_window = y[mask]

        if y_window.size > 0:
            # Use center of window and quantile of y as baseline point
            baseline_x.append(x_start + window_size / 2)
            baseline_y.append(np.quantile(y_window, quantile_value, method='hazen'))

        x_start += step_size

    # Interpolate baseline using PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
    f_interp = PchipInterpolator(baseline_x, baseline_y, extrapolate=True)
    baseline_curve = f_interp(x)

    # Subtract interpolated baseline from original signal
    adjusted_signal = y - baseline_curve
    return adjusted_signal
