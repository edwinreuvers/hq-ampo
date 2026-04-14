# %% [markdown]
# ---
# title: AMPO-scaling factors were successfully obtained
# ---

# %% [markdown]
"""
The analysis of this page corresponds to the section title 'AMPO-scaling
factors factors were successfully obtained'.

Specifically, the following steps were taken:
          
-   Load the measured experimental data and compute a moving average of the
    measured net knee joint moment over an interval of 0.5s and take the
    maximal value.
-   Compute the percentage difference of the maximal net knee joint moment 
    (averaged over the 0.5s interval) between the two knee joint angles
    that were repeated. Throw away the lowest of the two.
-   Estimate the muscle parameter values (fmax, lceopt & lsee0) for each 
    participant individually. 
      
Custom functions used:
    
-   `hillmodel.force_eq(lmtc, gamma, muspar)`
    : Computes the isometric SEE force at each MTC length.
-   `stats.pdiff(value1, value2)`
    : Computes percent difference between two values
-   `stats.rmse(signal1, signal2)`
    : computes root-mean-square error between signals.
-   `signal_processing.moving_average(signal, dt, fs, axis)`
    : computes moving average.
        
"""

# %% [markdown]
"""
## Load packages & set directories
"""

#%% Load packages
import glob, os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, io
from pathlib import Path

# Paths
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import hillmodel, signal_processing, stats

plt.close('all')

# %% [markdown]
"""
## Load data
"""

#%% Load data
# Load standard parameter values
parFile = os.path.join(dataDir,'VAS_muspar.pkl')
muspar = pickle.load(open(parFile, 'rb'))

# Loop over participants and knee angles
PPs = [1,2,3,4,5,6]
nPP = len(PPs)
phi_set = np.full((nPP,11), np.nan)
mom_set = np.full((nPP,11), np.nan)
for iPP, PP in enumerate(PPs):
    # Build folder path
    folder_path = os.path.join(dataDir, 'dataExp', f'pp{PP:02d}', 'day3')
    # File pattern: starts with 'pp01_day3_mh' and ends with '.mat'
    file_pattern = os.path.join(folder_path, f'pp{PP:02d}_day3_mh*.mat')
    # Find all matching files
    mat_files = sorted(glob.glob(file_pattern))
    
    for iFile,filepath in enumerate(mat_files):
        mat = io.loadmat(filepath)
        cfg = mat["Cfg"]
        data = mat["Data"]
        
        # Extract metadata
        fs = cfg["SampleFrequency"][0][0][0][0]
        
        # Extract dynamometer data
        phi = data[:, 0] / 180 * np.pi  # [rad] knee angle
        tor = data[:, 1] # [Nm] net knee joint torque
        time = np.arange(len(phi)) / fs  #  [s]
        
        # Compute moving average of 500 ms
        phi_movmean = signal_processing.moving_average(phi, dt=0.5, fs=fs)
        mom_movmean = signal_processing.moving_average(tor, dt=0.5, fs=fs)
        iMax = np.argmax(mom_movmean)
        phi_set[iPP,iFile] = phi_movmean[iMax]
        mom_set[iPP,iFile] = mom_movmean[iMax]


# %% [markdown]
"""
## Compute fatigue measures and trow away lowest of repeated conditons
"""

#%% Compute fatigue measures and trow away lowest of repeated conditons
valid_mask = np.ones_like(mom_set, dtype=bool)  # start with all True

p_rep = []
for iPP, pp in enumerate(PPs):
    # Build folder path
    folder_path = os.path.join(dataDir, 'dataExp', f'pp{pp:02d}', 'day3')
    # File pattern: starts with 'pp01_day3_mh' and ends with '.mat'
    file_pattern = os.path.join(folder_path, f'pp{pp:02d}_day3_mh*.mat')
    # Find all matching files
    mat_files = sorted(glob.glob(file_pattern))
    
    # Extract condition and trial type (t1 or t2)
    trials = [os.path.basename(f).split('_')[3].split('.')[0] for f in mat_files]  # t1, t2
    
    iPosts = [i for i, t in enumerate(trials) if t == 't2']
    iPres = [i-1 for i in iPosts]
    
    if not np.allclose(phi_set[iPP,iPres],phi_set[iPP,iPosts],rtol=0.001):
        # We have a problem, probably not the same knee angle!
        breakpoint()
    
    # Compute percentage difference for all repeated trials
    for iPre, iPost in zip(iPres, iPosts):
        pdiff_val = stats.pdiff(mom_set[iPP, iPost], mom_set[iPP, iPre])
        p_rep.append(pdiff_val)
        
    # Mask the trial with lower torque
    for iPre, iPost in zip(iPres, iPosts):
        if mom_set[iPP, iPre] < mom_set[iPP, iPost]:
            valid_mask[iPP, iPre] = False
        else:
            valid_mask[iPP, iPost] = False
            
p_rep = np.array(p_rep)   
phi_exp_clean = phi_set[valid_mask].reshape(nPP, -1)
mom_exp_clean = mom_set[valid_mask].reshape(nPP, -1)

# %% [markdown]
"""
## Compute muscle parameters for each participant
"""

#%% Compute muscle parameters for each participant
def computeError(x,phi_exp,mom_exp,muspar):
    parms = muspar.copy()
    parms['fmax']       = x[0]
    parms['lce_opt']    = x[1]
    parms['lsee0']      = x[2]
       
    lmtc    = phi_exp*parms['A1']+parms['A0']
    fsee    = hillmodel.force_eq(lmtc,1,parms)[0]
    mom_mdl = fsee*parms['A1']
    
    return np.sum((mom_exp-mom_mdl)**2), mom_mdl

rmsd    = np.full((nPP), np.nan)
fmax    = np.full((nPP), np.nan)
lce_opt = np.full((nPP), np.nan)
lsee0   = np.full((nPP), np.nan)
for i,(phi_exp,mom_exp) in enumerate(zip(phi_exp_clean, mom_exp_clean)): 
    # Perform optimisation
    x0 = [tor.max()/muspar['A1'], 0.07, 0.18]
    fun = lambda x: computeError(x,phi_exp,mom_exp,muspar)[0]
    result = optimize.minimize(fun,x0,method='Nelder-Mead',bounds=((1e-6, None),(1e-6, None),(1e-6, None)),options={'disp': True,'maxiter':1e4})
    
    # Extract parms
    estpar = {}
    estpar['fmax'], estpar['lce_opt'], estpar['lsee0'] = result.x
    
    # Compute rmsd
    mom_mdl = computeError(result.x,phi_exp,mom_exp,muspar)[1]
    rmsd[i] = stats.rmse(mom_exp,mom_mdl)/(estpar['fmax']*muspar['A1'])*100
    
    # Store muscle parameter values
    fmax[i] = estpar['fmax'] # [N]
    lce_opt[i] = estpar['lce_opt']*100 # [cm]
    lsee0[i] = estpar['lsee0']*100 # [cm]

# Save muscle parameters to CSV
df = pd.DataFrame({
    'Participant': PPs,
    'fmax [N]': fmax,
    'lce_opt [cm]': lce_opt,
    'lsee0 [cm]': lsee0,
    'rmsd [%]': rmsd
})

# Save to CSV
csv_file = dataDir / 'isometric-measurements.csv'
df.to_csv(csv_file, index=False)   