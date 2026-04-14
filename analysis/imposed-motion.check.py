# %% [markdown]
# ---
# title: Analyse imposed knee joint movements
# ---

# %% [markdown]
"""
The analysis of this page corresponds to the section 'Knee joint movements
were succesfuly imposed'.

Specifically, the following steps were taken:

-   Load condition information (cycle frequency, fraction of time spent shortening) from CSV.
-   Loop over each participant and condition:
    -   Load preprocessed AMPO measurement data (day 4).
    -   Extract relevant columns: time, knee angle (phi), torque, EMG, phase.
    -   Segment the time and angle data into cycles, extension, and flexion phases.
    -   Calculate extension time, flexion time, maximum and minimum angles for each cycle.
    -   Compute the expected theoretical knee angle trajectory for each cycle.
    -   Compute root mean square difference (RMSD) between measured and theoretical trajectories.
-   Compute RMSD statistics across participants, conditions, and cycles.
-   Calculate differences between measured and expected extension/flexion times.
-   Calculate differences in range of motion (ROM) between measured and theoretical angles.

Custom functions used:

-   `helpers.segments_to_list(time, phase, segment_type, cycle_indices)`
    : Segments the continuous time series data into lists of individual cycles, extension, or flexion phases.
-   `trajectories.dino(time_vec, cf, fts, kje, phi_avg, acc, first_contr)` 
    : Computes knee angle over time for the dynamometer based on SSC parameters.
-   `stats.rmse(signal1, signal2)`
    : computes root-mean-square error between signals.
    
"""

# %% [markdown]
"""
## Load packages & set directories
"""

#%% Load packages
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

# Import custom functions for data processing
import helpers, stats, trajectories

# %% [markdown]
"""
## Compute variables for each participant and condition
"""

#%% Compute variables for each participant and condition
# Load conditions data from CSV file
data = pd.read_csv(os.path.join(dataDir, 'conditions.csv'), sep=',', header=None).to_numpy()

# Define condition names
condNames = ['0.40', '0.35C', '0.30', '0.25', '0.20', '0.35A', '0.35B', '0.35D', '0.35E']

# Extract condition data for cycle frequencies (cf) and flexion time (fts)
cf = data[1].astype(float)
fts = data[2].astype(float)

# Calculate the expected extension and flexion times per condition
TextCond = fts / cf  # Extension time
TflxCond = (1 - fts) / cf  # Flexion time

# Define maximum and minimum knee angles for conditions
phiMaxCond = 1.8  # Max knee angle (radians)
phiMinCond = 0.2  # Min knee angle (radians)

# List of participants
pps = [1, 2, 3, 4, 5, 6]

# Initialize matrices for storing processed data
Ncycli = 14  # Number of cycles
Text = np.nan * np.zeros((len(pps), len(condNames), Ncycli))
Tflx = np.nan * np.zeros((len(pps), len(condNames), Ncycli))
phiMax = np.nan * np.zeros((len(pps), len(condNames), Ncycli))
phiMin = np.nan * np.zeros((len(pps), len(condNames), Ncycli))
rmsd = np.nan * np.zeros((len(pps), len(condNames), Ncycli))

day = 4  # Day of experiment
trial = 1  # Trial number

for iPP, pp in enumerate(pps):
    # Define directory for each participant's preprocessed data
    dataDirPP = os.path.join(dataDir, 'dataExp', f'pp{pp:02d}', 'AMPO measurements', '')
    
    for iCond, condName in enumerate(condNames):
        # Define filename for each condition
        filename = f'pp{pp:02d}_{condName}_t1'
        
        # Load the preprocessed data
        df = pd.read_csv(os.path.join(dataDirPP, filename + '.csv'))
        data = df.to_numpy()
        
        # Extract columns from the data
        time, phi, _, torqueComp, _, _, emgQuad, emgHams, phase = data.T[:9]

        # Process the extension and flexion times using the custom function
        mTimeExt = helpers.segments_to_list(time, phase, 'ext', np.arange(0, 14))
        mTimeFlx = helpers.segments_to_list(time, phase, 'flx', np.arange(0, 14))
        
        mTime = helpers.segments_to_list(time, phase, 'cycle', np.arange(0, 14))
        mPhi = helpers.segments_to_list(phi, phase, 'cycle', np.arange(0, 14))
        mPhiExt = helpers.segments_to_list(phi, phase, 'ext', np.arange(0, 14))
        
        # Calculate extension time, flexion time, max and min angles for each cycle
        for iCycle in range(Ncycli):
            Text[iPP, iCond, iCycle] = mTimeExt[iCycle][-1] - mTimeExt[iCycle][0]
            Tflx[iPP, iCond, iCycle] = mTimeFlx[iCycle][-1] - mTimeFlx[iCycle][0]
            phiMax[iPP, iCond, iCycle] = mPhiExt[iCycle].max()
            phiMin[iPP, iCond, iCycle] = mPhiExt[iCycle].min()
            
            phiTheor = trajectories.dino(mTime[iCycle]-mTime[iCycle][0],cf[iCond],fts[iCond],kje=1.6,phi_avg=1.0,acc=50,first_contr='E')[0]
            rmsd[iPP, iCond, iCycle] = stats.rmse(mPhi[iCycle],phiTheor)

# Root mean square differences
rmsd_mean = f"{rmsd.mean()*1e3:0.0f}" # [mrad] 
rmsd_std  = f"{rmsd.std()*1e3:0.0f}"  # [mrad] 

# Output ROM results
print(f"RMSD is: {rmsd_mean} ± {rmsd_std} mrad")

# Flexion and Extension Time Differences
# Create matrices for the condition's expected times
a = np.repeat([TextCond.T], 6, axis=0)  # Repeat for each participant
b = a[:, :, None]
TextCond = np.repeat(b, Ncycli, axis=2)

a = np.repeat([TflxCond.T], 6, axis=0)
b = a[:, :, None]
TflxCond = np.repeat(b, Ncycli, axis=2)

# Calculate the mean and standard deviation of time differences
Text_mean = f"{(Text - TextCond).mean() * 1e3:0.1f}"  # [ms] mean of extension time difference
Text_std = f"{(Text - TextCond).std() * 1e3:0.1f}"  # [ms]  std of extension time difference
Tflx_mean = f"{(Tflx - TflxCond).mean() * 1e3:0.1f}"  # [ms]  mean of flexion time difference
Tflx_std = f"{(Tflx - TflxCond).std() * 1e3:0.1f}"  # [ms] std of flexion time difference

# Output the results
print(f"Difference in extension time (desired - measured) is: {Text_mean} ± {Text_std} ms")
print(f"Difference in flexion time (desired - measured) is: {Tflx_mean} ± {Tflx_std} ms")

# Range of Motion (ROM) Calculation
# Calculate ROM (difference between max and min angles) for each cycle
rom_mean = f"{((phiMax - phiMin) - (phiMaxCond - phiMinCond)).mean() * 1e3:0.1f}" # [mrad] ROM mean
rom_std  = f"{((phiMax - phiMin) - (phiMaxCond - phiMinCond)).std() * 1e3:0.1f}"  # [mrad] ROM std

# Output ROM results
print(f"Difference in range of motion (desired - measured) is: {rom_mean} ± {rom_std} mrad")
