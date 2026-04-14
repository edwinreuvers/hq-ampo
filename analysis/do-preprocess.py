# %% [markdown]
# ---
# title: Preprocess data
# ---

# %% [markdown]
"""
On this page all pre-processing steps of the measure data are shown.

Specifically, it performs the following steps for each participant, day, 
trial, and condition:

-   Computes MVC (maximum voluntary contraction) values for quadriceps and hamstrings.
-   Loads and synchronizes measurement data from dynamometer ('dino') and EMG ('stil') measurement systems.
-   Detects individual cycles in knee joint angle signals and segments moment and EMG data accordingly.
-   Computes cycle-domain moment and applies passive moment compensation.
-   Reconstructs knee jont moment signals in the original time domain.
-   Computes phase within each cycle.
-   Preprocesses EMG signals: bandpass filtering, rectification via Hilbert 
    transform, smoothing, and normalisation to %MVC.
-   Optionally plots raw vs. compensated moment over time for inspection.
-   Stores all processed signals (knee joint moment, compensated moment, EMG, phase) 
    in a pandas DataFrame.
-   Saves the processed data to CSV for further analysis.

Custom functions used from preprocess module:

-   `build_filepaths(dataDir, pp, day, cond, trial)`
    : constructs file paths for MAT and CSV files.
-   `load_dino_data(filepath, pp)`
    : loads dynamometer data.
-   `load_stil_data(filepath, dino)`
    : loads EMG data and synchronizes with dynamometer data.
-   `detect_cycles(phi, fs, t_cycle, n_cycles)`
    : identifies cycle start and end points.
-   `cut_data(peaks_sf, peaks_se, dino, stil)`
    : segments the signals into cycles.
-   `compute_cycle_mom(phi, mom, time, peaks_sf, peaks_se)` 
    : computes knee joint moment per cycle.
-   `compensate_passive_mom(mom_cycles, pp, cond, trial)`
    : removes passive knee joint moment contributions.
-   `reconstruct_time_signal(time_cycles, mom_cycles, time_orig)`
    : interpolates cycle-domain data back to continuous time.
-   `compute_phase(peaks_sf, peaks_se)`
    : calculates phase within each cycle.

Custom functions used from signal_processing module:
 
-   `xcorr(signal1, signal2)`
    : computes cross-correlation for time alignment.
-   `moving_average(signal, dt, fs, axis)`
    : computes moving average.

Custom functions used from emg_processing module:

-   `preprocess_emg(emg, fs)`
    : filters raw EMG.
    
Custom functions used from stats module:
    
-   `rmse(signal1, signal2)`
    : computes root-mean-square error between signals.
"""

# %% [markdown]
"""
## Load packages & set directories
"""

#%% Load packages
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
plt.close('all')

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import preprocess, stats, emg_processing, signal_processing

# %% [markdown]
"""
## Do preprocessing
"""

# %% Do preprocessing
#| eval: false
do_plot = False
day = 4
trial = 2
pp = 1

for pp in [1,2,3,4,5,6]:
    # First compute MVC values
    filepaths = [os.path.join(dataDir,"dataExp",f"pp{pp:02d}",f"day{day}",f"pp{pp:02d}_day{day}_mvc_quad_t{trial}.csv") for trial in [1,2]]
    emg_mvc_quad = np.max(emg_processing.compute_mvc(filepaths)[:,0])
    filepaths = [os.path.join(dataDir,"dataExp",f"pp{pp:02d}",f"day{day}",f"pp{pp:02d}_day{day}_mvc_hams_t{trial}.csv") for trial in [1,2]]
    emg_mvc_hams = np.max(emg_processing.compute_mvc(filepaths)[:,1])
    
    for cond in [pp,pp+1]:
        print(pp)
        print(cond)
        # Filepaths
        filepath_mat, filepath_csv = preprocess.build_filepaths(
            dataDir, pp, day, cond, trial
        )
        
        # Load data
        dino = preprocess.load_dino_data(filepath_mat, pp)
        stil = preprocess.load_stil_data(filepath_csv, dino)
        
        # Check time-sync
        if stil is not None:
            xcorr,lags = signal_processing.xcorr(dino["phi"], stil["phi"])
            iMax = np.argmax(xcorr)  # Find the index of maximum correlation
            iShift = lags[iMax]  # Determine the time shift in samples                                          
            if iShift != 0:
                # Someting is wrong!
                breakpoint()  
        
        # Cycle detection
        peaks_sf, peaks_se = preprocess.detect_cycles(
            dino["phi"], dino["fs"],
            dino["t_cycle"], dino["n_cycles"]
        )
        
        # Cut data
        dino,stil,peaks_sf,peaks_se = preprocess.cut_data(peaks_sf,peaks_se,dino,stil)
        
        # Cycle-domain knee joint moment
        mom_cycles, time_cycles = preprocess.compute_cycle_mom(
            dino["phi"], dino["mom"], dino["time"],
            peaks_sf, peaks_se
        )
        
        # Passive compensation
        mom_comp_cycles = preprocess.compensate_passive_mom(
            mom_cycles, pp, cond, trial
        )
        
        # Interpolate back to time
        mom_comp = preprocess.reconstruct_time_signal(
            time_cycles, mom_comp_cycles, dino["time"]
        )
        
        # Now we have a few nan's, but thats no big deal..
        if np.sum(np.isnan(mom_comp)) > 100:
            breakpoint()
        else:
            mom_comp = np.interp(np.arange(len(mom_comp)), np.flatnonzero(~np.isnan(mom_comp)), mom_comp[~np.isnan(mom_comp)])
        
        # Phase
        phase = preprocess.compute_phase(
            peaks_sf, peaks_se
        )
        
        # Interpolate back to time original signal
        # Just as a test..
        mom_test = preprocess.reconstruct_time_signal(
            time_cycles, mom_cycles, dino["time"]
        )
        if stats.rmse(mom_test,dino['mom']) > 0.1:
            # something must have gone wrong!
            breakpoint()
        
        # EMG envelopes
        if stil is not None:
            # Preprocess EMG (bandpass)
            emg_filt = emg_processing.preprocess_emg(stil["emg"], fs=dino["fs"])
            # Rectify EMG using Hilbert transform
            emg_rect = np.abs(signal.hilbert(emg_filt,axis=0))
            # Smooth rectified EMG
            sremg = signal_processing.moving_average(emg_rect, dt=0.1, fs=dino["fs"], axis=0)
            # Normalise: express in %MVC
            emg_mvc = np.array([emg_mvc_quad, emg_mvc_hams])
            sremg = sremg/emg_mvc
        
        # Plot (optional)
        if do_plot:
            plt.figure()
            plt.plot(dino["time"], dino["mom"], label="Measure moment")
            plt.plot(dino["time"], mom_comp, label="Compensated moment (Mmus)")
            plt.legend()
            plt.xlabel("Time [s]")
            plt.ylabel("Knee joint moment [Nm]")
            plt.title("Knee joint moment with and without removing passive contribution")
            plt.show()
        
        # Create pandas df
        df = pd.DataFrame({
            'time [s]': dino["time"],
            'phi [rad]': dino["phi"],
            'mom [Nm]': dino["mom"],
            'mom_comp [Nm]': mom_comp,
            'emg_quad [mV]': stil["emg"][:, 0] if stil is not None else np.full_like(dino["time"], np.nan),
            'emg_hams [mV]': stil["emg"][:, 1] if stil is not None else np.full_like(dino["time"], np.nan),
            'sremg_quad [%]': sremg[:, 0] if stil is not None else np.full_like(dino["time"], np.nan),
            'sremg_hams [%]': sremg[:, 1] if stil is not None else np.full_like(dino["time"], np.nan),
            'phase': phase
        })
        
        # Save to CSV
        condition = ['0.40','0.35C','0.30','0.25','0.20','0.35A','0.35B','0.35D','0.35E'][cond-1]
        fname = f"pp{pp:02d}_{condition}_t{trial}"
        filepath = os.path.join(dataDir, "dataExp", f"pp{pp:02d}", "AMPO measurements", fname +".csv")
        df.to_csv(filepath, index=False)


