"""
Utilities for processing and analysing cyclic knee joint kinematics & kinetic data.

This module provides functions for loading, segmenting, and analysing 
dynamometer and EMG data from repeated knee flexion-extension cycles. 
It includes tools for:

- Constructing file paths for MAT and CSV data files.
- Loading dynamometer (.mat) and EMG (.csv) data.
- Detecting cycles based on flexion ('sf') and extension ('se') peaks.
- Cutting and aligning data to cycles.
- Aggregating and interpolating signals in cycle or angle domain.
- Normalizing and compensating knee joint moments.
- Reconstructing time-domain signals from cycle-domain data.
- Labeling movement phases for segmentation.

The functions handle participant-specific corrections, downsampling, 
and passive moment compensation where required. Data is typically returned 
as NumPy arrays or dictionaries for further analysis.
"""

import os
import numpy as np
from scipy import signal
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# %% Path utilities
def build_filepaths(data_dir, pp, day, cond, trial):
    """
    Construct filepaths for MAT and CSV files.

    Parameters
    ----------
    data_dir : str
        Directory containing the raw data.
    pp : int
        Participant ID.
    day : int
        Day of the experiment.
    cond : int
        Condition number of the experiment.
    trial : int
        Trial number.

    Returns
    -------
    tuple
        A tuple containing:
        - filepath_mat (str): Path to the MAT file.
        - filepath_csv (str): Path to the CSV file.
    """
    base = os.path.join(
        data_dir, "dataExp", f"pp{pp:02d}", f"day{day}"
    )

    fname = f"pp{pp:02d}_day{day}_cond{cond:02d}_t{trial}"

    filepath_mat = os.path.join(base, f"{fname}.mat")
    filepath_csv = os.path.join(base, f"{fname}.csv")

    return filepath_mat, filepath_csv


# %% Loading functions
def load_dino_data(filepath, pp):
    """
    Load dynamometer data from a .mat file.

    This function extracts relevant data from a .mat file, such as the knee angle 
    ('phi'), moment ('mom'), and metadata. 

    Parameters
    ----------
    filepath : str
        Path to the .mat file containing dynamometer data.
    pp : int
        Participant ID. One participant requires adjustments to their data.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - 'time' : ndarray
            Array of time values corresponding to the samples (in seconds).
        - 'phi' : ndarray
            Array of knee angle values (in radians).
        - 'mom' : ndarray
            Array of knee joint moment (in Nm).
        - 'fs' : float
            Sample frequency (in Hz).
        - 't_cycle' : float
            Duration of one cycle (in seconds).
        - 'n_cycles' : int
            Number of cycles recorded.
    """
    # Load data
    mat = loadmat(filepath)

    cfg = mat["Cfg"]
    data = mat["Data"]

    # Extract metadata
    fs = cfg["SampleFrequency"][0][0][0][0]
    t_cycle = abs(cfg["CyclusDuration"][0][0][0][0] / 1000)
    n_cycles = cfg["NrCycli"][0][0][0][0]

    # Extract dynamometer data
    phi = data[:, 0] / 180 * np.pi  # Convert knee angle from degrees to radians
    mom = data[:, 1] # [Nm]
    time = np.arange(len(phi)) / fs  # Create time vector based on sample frequency

    # Participant-specific correction for participant 5
    if pp == 5:
        phi += (1.8 - np.max(phi))  # Correction to the knee angle

    return {
        "time": time,
        "phi": phi,
        "mom": mom,
        "fs": fs,
        "t_cycle": t_cycle,
        "n_cycles": n_cycles,
    }


def load_stil_data(csv_path, dino, do_trim=True):
    """
    Load and downsample EMG data from a CSV file.

    This function loads EMG data from a CSV file, performs anti-aliasing filtering, 
    and downsamples it to a target sample frequency. It also returns the corresponding 
    time and knee angle data.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the EMG data.
    dino : dict
        Dictionary with the dynamometer data (see function above!)

    Returns
    -------
    dict or None
        If the file is found and loaded, returns a dictionary with the following keys:
        - 'time' : ndarray
            Array of time values corresponding to the samples (in seconds).
        - 'phi' : ndarray
            Array of knee angle values (in radians).
        - 'emg' : ndarray
            Array of EMG signals (each channel corresponds to one column).
        - 'fs' : float
            Target sample frequency after downsampling.
        
        Returns None if the file is not found or there is an error loading the data.
    """
    try:
        data = np.loadtxt(csv_path, delimiter=",")
        
        fs = 4000  # Original sampling frequency of the EMG data (Hz)
        phi = data[:, 0]  # Knee angle in radians
        emg = data[:, [3, 5]]  # Raw EMG signals (e.g., channels 3 and 5)
        time = np.arange(len(emg)) / fs  # Time axis for the raw EMG data
        
        # Anti-aliasing filter design
        fs_target = dino["fs"]
        b, a = signal.butter(1, fs_target / 2, fs=fs, btype="low")
        emg_filt = signal.filtfilt(b, a, emg, axis=0)

        # Downsample the data to the target sample frequency
        time_ds = np.arange(0, time[-1], 1 / fs_target)
        phi_ds = np.interp(time_ds, time, phi)
        emg_ds = np.array([np.interp(time_ds, time, x) for x in emg_filt.T]).T
        
        # Trim data
        if do_trim:
            n_target = len(dino["phi"])
            time_ds = time_ds[0:n_target]
            phi_ds = phi_ds[0:n_target]
            emg_ds = emg_ds[0:n_target]
        
        return {
            "time": time_ds,
            "phi": phi_ds,
            "emg": emg_ds,
            "fs": fs_target,
        }

    except OSError:
        # Return None if the CSV file cannot be found or loaded
        return None

# %% Cycle detection
def detect_cycles(phi, fs, t_cycle, n_cycles):
    """
    Detect indices of flexion (sf) and extension (se) peaks.

    Parameters
    ----------
    phi : ndarray
        Knee angle values (in radians).
    fs : float
        Sample frequency (in Hz).
    t_cycle : float
        Duration of one cycle (in seconds).
    n_cycles : int
        Number of cycles to detect.

    Returns
    -------
    peaks_sf : ndarray
        Indices of flexion peaks.
    peaks_se : ndarray
        Indices of extension peaks.
    """
    peaks_sf, _ = find_peaks(-phi, distance=0.9 * fs * t_cycle)
    peaks_se, _ = find_peaks(phi, distance=0.9 * fs * t_cycle)

    if peaks_sf[0] > 0.9 * fs * t_cycle:
        peaks_sf = np.insert(peaks_sf, 0, 0)

    return peaks_sf[:n_cycles + 1], peaks_se[:n_cycles]


# %% Cut data
def cut_data(peaks_sf, peaks_se, data_dino, data_stil):
    """
    Cut dynamometer and EMG data based on detected cycle peaks.

    Parameters
    ----------
    peaks_sf : ndarray
        Indices of flexion peaks.
    peaks_se : ndarray
        Indices of extension peaks.
    data_dino : dict
        Dictionary containing the dynamometer data (including 'time', 'phi', 'mom').
    data_stil : dict
        Dictionary containing the EMG data (including 'time', 'phi', 'emg').

    Returns
    -------
    data_dino : dict
        Cut dynamometer data.
    data_stil : dict
        Cut EMG data.
    peaks_sf : ndarray
        Adjusted indices of flexion peaks.
    peaks_se : ndarray
        Adjusted indices of extension peaks.
    """
    # Apply the shift to the datasets (if needed)
    data_dino['time']   = data_dino['time'][peaks_sf[0]:peaks_sf[-1]+1] - data_dino['time'][peaks_sf[0]]
    data_dino['phi']    = data_dino['phi'][peaks_sf[0]:peaks_sf[-1]+1]
    data_dino['mom']    = data_dino['mom'][peaks_sf[0]:peaks_sf[-1]+1]
    if data_stil is not None:
        data_stil['time']   = data_stil['time'][peaks_sf[0]:peaks_sf[-1]+1]
        data_stil['phi']    = data_stil['phi'][peaks_sf[0]:peaks_sf[-1]+1]
        data_stil['emg']    = data_stil['emg'][peaks_sf[0]:peaks_sf[-1]+1]
    
    peaks_se = peaks_se - peaks_sf[0]
    peaks_sf = peaks_sf - peaks_sf[0]
    
    return data_dino, data_stil, peaks_sf, peaks_se


# %% Helper
def consolidate(phi, data):
    """
    Aggregate values for identical angles.

    Parameters
    ----------
    phi : ndarray
        Knee angle values (in radians).
    data : ndarray
        Data values to aggregate.

    Returns
    -------
    phi_unique : ndarray
        Unique knee angle values.
    data_out : ndarray
        Aggregated data values.
    """
    phi_unique = np.unique(phi)
    data_out = np.array([
        np.nanmean(data[phi == p]) for p in phi_unique
    ])
    return phi_unique, data_out


# %% Cycle-domain mom
def compute_cycle_mom(phi, mom, time, peaks_sf, peaks_se):
    """
    Normalize knee joint moment per cycle in angle domain.

    Parameters
    ----------
    phi : ndarray
        Knee joint angle (in radians).
    mom : ndarray
        Knee joint moment (in Nm).
    time : ndarray
        Time values corresponding to the samples (in seconds).
    peaks_sf : ndarray
        Indices of flexion peaks.
    peaks_se : ndarray
        Indices of extension peaks.

    Returns
    -------
    mom_cycles : ndarray
        Knee joint moment per cycle.
    time_cycles : ndarray
        Time values corresponding to each cycle.
    """
    mom_cycles = []
    time_cycles = []

    phi_f_max = np.sort(phi[peaks_sf[:-1]])  # full flexion angles
    phi_e_max = np.sort(phi[peaks_se[:-1]])  # full extension angles
    phi_f_grid = np.concatenate([phi_f_max[:-1], np.linspace(phi_f_max[-1], phi_e_max[0], 5003 - len(phi_f_max) - len(phi_e_max)), phi_e_max[1:]])
    phi_e_grid = np.concatenate([phi_e_max[::-1][1:], np.linspace(phi_e_max[-1], phi_f_max[-1], 5003 - len(phi_f_max) - len(phi_e_max)), phi_f_max[:-1][::-1]])
    
    for i in range(len(peaks_sf)-1):
        # Extract value of current cycle
        phi_f = phi[peaks_sf[i]:peaks_se[i]]
        phi_e = phi[peaks_se[i]:peaks_sf[i + 1]]
        M_f = mom[peaks_sf[i]:peaks_se[i]]
        M_e = mom[peaks_se[i]:peaks_sf[i + 1]]
        time_f = time[peaks_sf[i]:peaks_se[i]]
        time_e = time[peaks_se[i]:peaks_sf[i + 1]]
        
        # Interpolate for joint moment
        phi_cf, M_cf = consolidate(phi_f, M_f)
        phi_ce, M_ce = consolidate(phi_e, M_e)
        M_F = interp1d(phi_cf, M_cf, kind='linear', bounds_error=False, fill_value=np.nan)(phi_f_grid)
        M_E = interp1d(phi_ce, M_ce, kind='linear', bounds_error=False, fill_value=np.nan)(phi_e_grid)
        
        # Interpolate for time-axis
        phi_cf, time_cf = consolidate(phi_f, time_f)
        phi_ce, time_ce = consolidate(phi_e, time_e)
        time_F = interp1d(phi_cf, time_cf, kind='linear', bounds_error=False, fill_value=np.nan)(phi_f_grid)
        time_E = interp1d(phi_ce, time_ce, kind='linear', bounds_error=False, fill_value=np.nan)(phi_e_grid)
        
        # Append
        mom_cycles.append(np.concatenate([M_F, M_E[1:]]))
        time_cycles.append(np.concatenate([time_F, time_E[1:]]))
        
    return np.array(mom_cycles), np.array(time_cycles)


# %% Passive compensation
def compensate_passive_mom(Mn, pp, cond, trial):
    """
    Remove passive moment contribution.

    Parameters
    ----------
    Mn : ndarray
        Measured knee joint moment (including passive component).
    pp : int
        Participant ID.
    cond : int
        Condition number.
    trial : int
        Trial number.

    Returns
    -------
    ndarray
        Knee joint moment with passive component removed.
    """
    if pp == 2 and cond == 9 and trial == 1:
        idx = [1, 2, 3, 12, 13]
    elif pp == 4 and cond == 1 and trial == 1:
        idx = [2, 3, 11, 12, 13]
    else:
        idx = [1, 2, 3, 11, 12, 13]

    Mp = Mn[idx, :]
    return Mn - np.nanmean(Mp, axis=0)


# %% Back to time domain
def reconstruct_time_signal(time_cycles, data_cycles, time_target):
    """
    Convert cycle-domain data back to time domain.

    Parameters
    ----------
    time_cycles : ndarray
        Time values corresponding to each cycle.
    data_cycles : ndarray
        Data values corresponding to each cycle.
    time_target : ndarray
        Target time values for reconstruction.

    Returns
    -------
    ndarray
        Reconstructed data values at the target time points.
    """
    mask = ~np.isnan(time_cycles)

    t_flat = time_cycles[mask]
    d_flat = data_cycles[mask]

    t_unique, idx = np.unique(t_flat, return_inverse=True)

    d_mean = np.array([
        np.mean(d_flat[idx == i])
        for i in range(len(t_unique))
    ])

    return interp1d(
        t_unique, d_mean,
        bounds_error=False,
        fill_value=np.nan
    )(time_target)


# %% Phase labeling
def compute_phase(peaks_sf, peaks_se):
    """
    Label movement phases ('sf', 'se').

    Parameters
    ----------
    peaks_sf : ndarray
        Indices of flexion peaks.
    peaks_se : ndarray
        Indices of extension peaks.

    Returns
    -------
    phase : ndarray
        Array of phase labels ("sf", "se", "f" or "o").
    """
    phase = np.full(peaks_sf[-1]+1, "", dtype="<U5")

    for i in range(len(peaks_sf) - 1):
        phase[peaks_sf[i]:peaks_se[i]] = 'f'
        phase[peaks_se[i]:peaks_sf[i + 1]] = 'e'
        phase[peaks_sf[i]] = "sf"
        phase[peaks_se[i]] = "se"
    phase[-1] = "sf"
    
    return phase