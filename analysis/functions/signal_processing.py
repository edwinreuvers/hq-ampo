"""
Signal processing utilities for time-series analysis.

This module provides functions for basic signal operations, including:

- Moving average smoothing for signals of arbitrary dimensionality.
- Cross-correlation computation between two signals.
"""

import numpy as np

def moving_average(x, dt, fs, axis=-1):
    """
    Compute the moving average along a specified axis.

    This function applies a centered moving average filter using a
    rectangular window of duration `dt`. It works for arrays of any
    dimensionality by applying the filter along the specified axis.

    Parameters
    ----------
    x : ndarray
        Input array (can be 1D, 2D, 3D, ...).
    dt : float
        Window duration in seconds.
    fs : float
        Sampling frequency in Hz.
    axis : int, optional
        Axis along which to apply the moving average (default is -1).

    Returns
    -------
    y : ndarray
        Smoothed array with the same shape as `x`.

    Notes
    -----
    - The window length is computed as ``N = int(dt * fs)``.
    - A centered moving average is used.
    - Padding is applied using ``mode='edge'`` to preserve signal length.
    """
    x = np.asarray(x, dtype=float)
    N = int(dt * fs)

    if N < 1:
        raise ValueError("Window length N must be at least 1.")

    kernel = np.ones(N) / N

    def _movavg_1d(signal):
        signal_pad = np.pad(
            signal,
            (N // 2, N - 1 - N // 2),
            mode="edge"
        )
        return np.convolve(signal_pad, kernel, mode="valid")

    return np.apply_along_axis(_movavg_1d, axis, x)

def xcorr(x, y):
    """
    Compute the cross-correlation between two signals.

    Parameters
    ----------
    x : array-like
        First input signal (e.g., a time series or signal).
    y : array-like
        Second input signal.
    
    Returns
    -------
    xcorr : ndarray
        The cross-correlation between `x` and `y` based on the chosen `mode`.
    lags : ndarray
        The lag values corresponding to the cross-correlation values.
    """
    
    # Center the signals (zero mean)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    N = len(x)
    
    # Compute the cross-covariance
    xcov = np.correlate(x_centered, y_centered, mode='full')/N

    # Compute the cross-correlation
    xcorr = xcov / (np.std(x) * np.std(y))

    # Lag values range from -(N-1) to (N-1)
    lags = np.arange(-(N - 1), N)
    
    return xcorr, lags