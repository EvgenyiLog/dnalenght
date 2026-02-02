import numpy as np
import pywt


def wden(signal, threshold_rule='sqtwolog', threshold_type='s', noise_est_method='sln', level=1, wavelet='sym2'):
    """
    Denoise a signal using wavelet thresholding similar to MATLAB's wden function.

    Parameters:
      signal : array_like
          The input noisy signal.
      threshold_rule : str, optional
          Threshold selection rule. Currently supports 'sqtwolog' (universal threshold).
      threshold_type : str, optional
          Type of thresholding: 's' for soft, 'h' for hard.
      noise_est_method : str, optional
          Noise estimation method: 'sln' for level-dependent noise estimation.
      level : int, optional
          Level of wavelet decomposition.
      wavelet : str, optional
          Wavelet name (e.g. 'sym2').

    Returns:
      denoised_signal : ndarray
          The denoised signal.
    """
    # Convert threshold_type to pywt mode: 's' -> 'soft', 'h' -> 'hard'
    mode = 'soft' if threshold_type.lower() == 's' else 'hard'

    # Perform wavelet decomposition of the signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # coeffs[0] is the approximation coefficients; the rest are detail coefficients.

    # For level-dependent noise estimation ('sln'), calculate threshold for each detail level
    new_coeffs = [coeffs[0]]  # keep the approximation coefficients unaltered
    for i, d in enumerate(coeffs[1:], start=1):
        if noise_est_method.lower() == 'sln':
            # Estimate noise sigma for this level using the median absolute deviation
            sigma_i = np.median(np.abs(d - np.median(d))) / 0.6745
            # Universal threshold: sigma * sqrt(2*log(n)) where n is the length of the detail coeff.
            thresh = sigma_i * np.sqrt(2 * np.log(len(d)))
        else:
            # Alternatively, use a global noise estimation from the finest scale (first detail level)
            # Here we compute sigma once from the first detail coefficients
            sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
            thresh = sigma * np.sqrt(2 * np.log(len(d)))

        # Apply thresholding to the detail coefficients
        new_coeffs.append(pywt.threshold(d, thresh, mode=mode))

    # Reconstruct the signal using the thresholded coefficients
    denoised_signal = pywt.waverec(new_coeffs, wavelet)

    # In case the reconstructed signal is longer than the original (due to padding), trim it.
    denoised_signal = denoised_signal[:len(signal)]
    return denoised_signal
