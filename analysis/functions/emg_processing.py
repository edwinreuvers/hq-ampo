"""
This module provides functions for preprocessing electromyography (EMG) signals
and computing Maximum Voluntary Contraction (MVC) from CSV data files.

Functions
---------
preprocess_emg(emg, fc=[20, 499.999], fs=1000)
    Apply a bandpass filter to raw EMG data.

compute_mvc(filepaths, dt=0.5)
    Compute the MVC for each EMG channel from a list of CSV files.

Notes
-----
- EMG signals are typically filtered between 20 Hz and 500 Hz.
- Rectification is done using the Hilbert transform.
- Smoothing is applied via a moving average with configurable window duration.
- All functions assume input EMG signals are time x channels (2D arrays).
"""

import numpy as np
import scipy.signal as signal

import preprocess, signal_processing

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def preprocess_emg(emg, fc=[20, 499.999], fs=1000):
    """
    Preprocess EMG signal using bandpass filter

    Parameters
    ----------
    emg : np.ndarray
        Raw EMG data (time x channels).
    fc : list of float
        Bandpass filter cutoff frequencies (low, high) in Hz.
    fs : float
        Sampling frequency in Hz.


    Returns
    -------
    np.ndarray
        Smoothed rectified EMG signal.
    """
    # Bandpass filter
    b, a = signal.butter(1, fc, fs=fs, btype='bandpass')
    emg_filt = signal.filtfilt(b, a, emg, axis=0)

    return emg_filt

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_mvc(filepaths: list[str], dt: float = 0.5) -> np.ndarray:
    """
    Compute the Maximum Voluntary Contraction (MVC) for EMG signals from csv-files.

    Parameters
    ----------
    filepaths : list of str
        List of file paths to STIL EMG data files.
    dt : float, optional
        Time window (in seconds) for moving average smoothing. Default is 0.5 s.

    Returns
    -------
    np.ndarray
        Array of shape (n_files, n_channels) containing the maximum smoothed EMG
        amplitude for each channel in each file.
    """
    emg_mvc = []

    for filepath in filepaths:
        dino = {"fs": 1000}  # Sampling frequency [Hz]

        # Load EMG from STIL file
        stil_data = preprocess.load_stil_data(filepath, dino, do_trim=False)
        emg_signal = stil_data['emg']  # Raw EMG [mV]

        # Preprocess EMG (bandpass)
        emg_filt = preprocess_emg(emg_signal, fs=dino["fs"])

        # Rectify EMG using Hilbert transform
        emg_rect = np.abs(signal.hilbert(emg_filt, axis=0))

        # Smooth rectified EMG
        emg_smoothed = signal_processing.moving_average(emg_rect, dt=dt, fs=dino["fs"], axis=0)

        # Maximum value per channel
        emg_mvc.append(np.max(emg_smoothed, axis=0))

    return np.array(emg_mvc)