# %% [markdown]
# ---
# title: EMG and mechanical work
# ---

# %% [markdown]
"""
The analysis of this page corresponds to the sections 'Timing of m. quadriceps 
femoris activation was adequate' & Mechanical work was achieved almost 
exclusively during knee extension’.

Specifically, the following steps were taken:

-   Define participants and experimental conditions.
-   Load preprocessed AMPO measurement data for each participant, condition, and trial.
-   Segment EMG, torque, and knee angle data into cycles, extension, or flexion phases.
-   Cut EMG and torque data to a specific knee angle range (phi_range) for analysis.
-   Compute mean and standard deviation of quadriceps and hamstring EMG during extension and flexion.
-   Compute cross-correlation between quadriceps and hamstring EMG signals during extension.
-   Compute absolute mechanical work (Mmus) during flexion over the specified knee angle range.

Custom functions used:

-   `helpers.segments_to_list(data_array, phase_array, part, cycle_indices)`
    : Segments time series data (EMG, torque, or angle) into individual cycles or specific movement phases.
-   `helpers.data_cut(data_list, phi_list, phi_range)`
    : Restricts data to a given range of knee angles and returns the cut data for analysis.

"""

# %% [markdown]
"""
## Load packages & set directories
"""

#%% Load packages
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

# Import custom functions
import helpers

# %% [markdown]
"""
## Helper function to load and process data
"""

#%% Helper function to load and process data
def process_data_for_condition(pp, cond, part = 'cycle', phi_range=[0.45, 1.55], trial=1, iCycles=[5, 6, 7, 8, 9]):
    """
    Process data for a given participant (pp) and condition (cond) across multiple cycles.

    Parameters:
    -----------
    pp : int
        Participant ID.
    cond : str
        Condition identifier.
    trial : int, optional
        Trial number (default is 1).
    iCycles : list, optional
        List of cycles to process (default is [5, 6, 7, 8, 9]).

    Returns:
    --------
    emgQuadCut : list
        Processed EMG data for quadriceps.
    emgHamsCut : list
        Processed EMG data for hamstrings.
    """
    # Construct file path using pathlib
    dataDirPP = dataDir / 'dataExp' / f'pp{pp:02d}' / 'AMPO measurements'
    filename = f'pp{pp:02d}_{cond}_t{trial}'
    
    # Load the data
    df = pd.read_csv(dataDirPP / (filename + '.csv'))
    data = df.T.to_numpy()
    
    # Extract relevant data columns
    time, phi, _, torqueComp, _, _, emgQuad, emgHams, phase = data

    # Adjust cycles for specific conditions
    if pp == 3 and cond == '0.35E':
        iCycles = [6, 7, 8, 9, 10]  # For pp3 and cond 0.35E, adjust the cycle indices

    # Extract phase-specific data for flexion/extension
    mPhi        = helpers.segments_to_list(phi, phase, part, iCycles)
    mEMGquad    = helpers.segments_to_list(emgQuad, phase, part, iCycles)
    mEMGhams    = helpers.segments_to_list(emgHams, phase, part, iCycles)
    mTor        = helpers.segments_to_list(torqueComp, phase, part, iCycles)

    # Cut the EMG data within the specified range
    emg_quad_cut, _ = helpers.data_cut(mEMGquad, mPhi, phi_range)
    emg_hams_cut, _ = helpers.data_cut(mEMGhams, mPhi, phi_range)
    tor_cut, _      = helpers.data_cut(mTor, mPhi, phi_range) 

    return emg_quad_cut, emg_hams_cut, tor_cut

#%% Main loop to process all participants and conditions
def process_all_conditions(part = 'cycle', phi_range=[0.45, 1.55]):
    """
    Process all participants and conditions to calculate data over a certain range of knee angles.
    """
    pps = [1, 2, 3, 4, 5, 6]
    conds = ['0.40', '0.35C', '0.30', '0.25', '0.20', '0.35A', '0.35B', '0.35D', '0.35E']
    
    # Initialize lists to store the results
    emg_quad_tot = []
    emg_hams_tot = []
    tor_tot = []
    
    # Process each condition and participant
    for cond in conds:
        for pp in pps:
            emg_quad_cut, emg_hams_cut, tor_cut = process_data_for_condition(pp, cond, part, phi_range)

            # Append the processed EMG data to the total list
            emg_quad_tot.extend(emg_quad_cut)
            emg_hams_tot.extend(emg_hams_cut)
            tor_tot.extend(tor_cut)
            
    return emg_quad_tot, emg_hams_tot, tor_tot

# %% [markdown]
"""
## EMG during extension between phi 0.45-1.55
"""

#%% EMG during extension between phi 0.45-1.55
emg_quad_tot, emg_hams_tot = process_all_conditions(part='ext', phi_range=[0.45, 1.55])[0:2]

emg_quad_ext_mean = f"{np.concatenate(emg_quad_tot).astype(float).mean()*1e2:0.1f}"
emg_quad_ext_std  = f"{np.concatenate(emg_quad_tot).astype(float).std()*1e2:0.1f}"
emg_hams_ext_mean = f"{np.concatenate(emg_hams_tot).astype(float).mean()*1e2:0.1f}"
emg_hams_ext_std  = f"{np.concatenate(emg_hams_tot).astype(float).std()*1e2:0.1f}"

# Calculate and print the mean and standard deviation for both muscles
print("Quadriceps EMG during extension (between knee angle 0.45-1.55 rad): " +
      f"{emg_quad_ext_mean} ± {emg_quad_ext_std} %")
print("Hamsrings EMG during extension (between knee angle 0.45-1.55 rad): " +
      f"{emg_hams_ext_mean} ± {emg_hams_ext_std} %")

# %% [markdown]
"""
## Cross-correlationf EMGs during extension between phi 0.2-1.8
"""

#%% Cross-correlationf EMGs during extension between phi 0.2-1.8
# Process all conditions and participants with a different phi_range
emg_quad_tot, emg_hams_tot = process_all_conditions(part = 'ext', phi_range=[0.2, 1.8])[0:2]

# Vectorize cross-correlation calculation using numpy
xcorr = [
    np.corrcoef(np.array(x).astype(float), np.array(y).astype(float))[0, 1]
    for x, y in zip(emg_quad_tot, emg_hams_tot)
]

# nanmean because there is 1 participant 1 condition where EMG was not saved..
xcorr_mean = f"{np.nanmean(xcorr):0.2f}"
xcorr_std  = f"{np.nanstd(xcorr):0.2f}"

# Calculate and print the mean and standard deviation of cross-correlation values
print("Cross-correlation between quad & hams EMG is: " + 
      f"{xcorr_mean} ± {xcorr_std} %")

# %% [markdown]
"""
## EMG during flexion between phi 0.45-1.55
"""

#%% EMG during flexion between phi 0.45-1.55
# Process all conditions and participants
emg_quad_tot, emg_hams_tot = process_all_conditions(part='flx', phi_range=[0.45, 1.55])[0:2]

emg_quad_flx_mean = f"{np.concatenate(emg_quad_tot).astype(float).mean()*1e2:0.1f}"
emg_quad_flx_std  = f"{np.concatenate(emg_quad_tot).astype(float).std()*1e2:0.1f}"
emg_hams_flx_mean = f"{np.concatenate(emg_hams_tot).astype(float).mean()*1e2:0.1f}"
emg_hams_flx_std  = f"{np.concatenate(emg_hams_tot).astype(float).std()*1e2:0.1f}"

# Calculate and print the mean and standard deviation for both muscles
print("Quadriceps EMG during flexion (between knee angle 0.45-1.55 rad): " +
      f"{emg_quad_flx_mean} ± {emg_quad_flx_std} %")
print("Hamsrings EMG during flexion (between knee angle 0.45-1.55 rad): " +
      f"{emg_hams_flx_mean} ± {emg_hams_flx_std} %")

# %% [markdown]
"""
## Absolute Mmus during flexion between phi 0.45-1.55
"""

#%% Absolute Mmus during flexion between phi 0.45-1.55
# Process all conditions and participants
Mmus_tot = process_all_conditions(part='flx', phi_range=[0.45, 1.55])[2]

Mmus_mean = f"{np.abs(np.concatenate(Mmus_tot).astype(float)).mean():0.1f}"
Mmus_std = f"{np.abs(np.concatenate(Mmus_tot).astype(float)).mean():0.1f}"

# Calculate and print the mean and standard deviation for Mmus
print("Mmus flexion (between knee angle 0.45-1.55 rad): " +
      f"{Mmus_mean} ± {Mmus_std} Nm")