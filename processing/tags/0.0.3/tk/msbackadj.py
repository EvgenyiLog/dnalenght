import numpy as np
import scipy.signal as signal


def msbackadj(x, y, window_size=140, step_size=40, quantile_value=0.1):
    """
    Baseline correction function similar to MATLAB's msbackadj.

    Parameters:
    x : array-like
        Independent variable (e.g., m/z values)
    y : array-like
        Intensity values to correct
    window_size : int, optional
        Size of the window used for baseline estimation (default: 140)
    step_size : int, optional
        Step size for moving window (default: 40)
    quantile_value : float, optional
        Quantile value for baseline estimation (default: 0.1, or 10%)
    show_plot : bool, optional
        Whether to show the baseline correction plot (default: False)

    Returns:
    y_adj : array-like
        Baseline-corrected intensity values
    """
    x = np.array(x)
    y = np.array(y)
    baseline = np.zeros_like(y)

    for start in range(0, len(x), step_size):
        end = min(start + window_size, len(x))
        window = y[start:end]
        baseline[start:end] = np.quantile(window, quantile_value)

    # Smooth the estimated baseline
    baseline = signal.savgol_filter(baseline, window_length=window_size // 2 * 2 + 1, polyorder=2)

    # Adjust the signal
    y_adj = y - baseline
    y_adj[y_adj < 0] = 0  # Ensure non-negative values

    return y_adj
