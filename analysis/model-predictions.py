# %% [markdown]
# ---
# title: Model predictions
# excecute:
#   eval: false
# ---

# %% [markdown]
"""
On this page all model predictions of the influence of SSC parameters on the
maximally attainable AMPO are derived.

Specifically, the maximally attainable AMPO is predicted for three situations:
          
-   A large number of knee joint movements that can be imposed by the knee
    dynamometer used. That means, a constant knee joint acceleration around
    transition from flexion-to-extension and extension-to-flexion. Due 
    to limits set on knee joint acceleration, not all combinations of cycle
    frequency, FTS and knee joint excursion are possible. Prediction are
    only derived for feasible combinations.
-   For the conditons that are investigated experimentally.
-   A large number of knee joint movement different constant knee angular 
    velocities during flexion and extension. Because knee angular velocity 
    was constant during flexion and extension, an unlimited set of 
    combinations of cycle frequency, FTS and knee joint excursion could be 
    investigated. 
"""

# %% [markdown]
"""
## Load packages & set directories
"""

#%% Imports
import os, sys, pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

# Custom imports
import derive_predictions, trajectories, stats

# %% [markdown]
"""
## Load muscle parameters
"""

#%% Load Parameters
parFile = os.path.join(dataDir,'VAS_muspar.pkl')
muspar = pickle.load(open(parFile, 'rb'))
lmtcOpt = (1+0.04)*muspar['lsee0']+muspar['lce_opt']

# %% [markdown]
"""
## SSC with knee dynamometer trajectory
"""

#%% SSC with knee dynamometer trajectory
# Setting used for dynamometer experiments
acc = 50 # [rad/s^2]
phi_avg = 1.0 # [rad]

# Derive predictions
for kje in [1.6]: # [rad]
    maxcf = (0.5)*acc**(1/2)/2/kje**(1/2) # maximum CF that can be imposed given acc and amp
    cf_set = np.arange(np.floor(maxcf*100)/100,0.33,-0.02)[::-1] # range of CF that we can investigate
    fts_max = 1 - (2*cf_set*kje**(1/2))/acc**(1/2) # maximum fts that can be imposed given acc, amp and cf
    fts_wide_range = np.hstack([1-stats.floor(fts_max,3), stats.floor(fts_max[::-1],3)]) # maximum range of cf

    for iCf,cf in enumerate(cf_set): # [Hz]
        fts_set = fts_wide_range[iCf:-iCf]
        for fts in fts_set:  
            if cf < 0.9 and cf > 0.38:
                Wmech, y = derive_predictions.optimise_stim(1,cf,fts,kje,phi_avg,acc,muspar,trajectories.dino)
                time, phi, stim, gamma, lcerel, q, lmtc, lsee, lpee, fisomrel, fsee, fpee, fce, fcerel, vcerel = y
                # Save data
                data = np.vstack((time,phi,stim,fsee,gamma,lcerel,q,lmtc,fisomrel)).T
                filepath = os.path.join(dataDir / 'simsExp' / 'contour', f'kje{kje:{"0.2f"}}rad_cf{cf:{"0.2f"}}Hz_fts{fts:{"0.3f"}}.csv')
                pd.DataFrame(data).to_csv(filepath,index=False,header=['Time [s]','Phi [rad]',
                    'STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]','q [ ]','Lmtc [m]','fisomrel [ ]'])

# %% [markdown]
"""
## SSC of specific experimental conditions
"""

#%% SSC of specific experimental conditions
# # Load conditions
# data = pd.read_csv(dataDir / 'Conditions.csv', sep=',', header=None, dtype=str).to_numpy()
# cond_names = data[0]
# cf_set  = data[1].astype(float) # [Hz]
# fts_set = data[2].astype(float) # []
# ampo_set= data[3].astype(float) # [1/s]

# # Setting used for dynamometer experiments
# kje = 1.6 # [rad]
# acc = 50 # [rad/s^2]
# phi_avg = 1.0 # [rad]

# # Derive predictions
# for cond_name,cf,fts in zip(cond_names, cf_set, fts_set):
#     Wmech, y = derive_predictions.optimise_stim(1,cf,fts,kje,phi_avg,acc,muspar,trajectories.dino)
#     time, phi, stim, gamma, lcerel, q, lmtc, lsee, lpee, fisomrel, fsee, fpee, fce, fcerel, vcerel = y
#     print(Wmech*cf)
    
#     # Save data
#     data = np.vstack((time,phi,stim,fsee,gamma,lcerel,q,lmtc,fisomrel)).T
#     filepath = os.path.join(dataDir / 'simsExp', f'cond_{cond_name}.csv')    
#     pd.DataFrame(data).to_csv(filepath,index=False,header=['Time [s]','Phi [rad]',
#         'STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]','q [ ]','Lmtc [m]','fisomrel [ ]'])


# %% [markdown]
"""
## SSC of specific experimental conditions - examine sensivity for lceopt
"""

#%% SSC of specific experimental conditions - examine sensivity for lceopt
# # Scale lce_opt
# muspar['A0'] -= 0.1*muspar['lce_opt'] # change such that optimise knee joint angle does not change!
# # muspar['lsee0'] += 0.1*muspar['lce_opt'] # change opt MTC length does not change
# muspar['lce_opt'] *= 0.9

# # Load conditions
# data = pd.read_csv(dataDir / 'Conditions.csv', sep=',', header=None, dtype=str).to_numpy()
# cond_names = data[0]
# cf_set  = data[1].astype(float) # [Hz]
# fts_set = data[2].astype(float) # []
# ampo_set= data[3].astype(float) # [1/s]

# # Setting used for dynamometer experiments
# kje = 1.6 # [rad]
# acc = 50 # [rad/s^2]
# phi_avg = 1.0 # [rad]

# # Derive predictions
# for cond_name,cf,fts in zip(cond_names, cf_set, fts_set):
#     Wmech, y = derive_predictions.optimise_stim(1,cf,fts,kje,phi_avg,acc,muspar,trajectories.dino)
#     time, phi, stim, gamma, lcerel, q, lmtc, lsee, lpee, fisomrel, fsee, fpee, fce, fcerel, vcerel = y
#     print(Wmech*cf)
    
#     # Save data
#     data = np.vstack((time,phi,stim,fsee,gamma,lcerel,q,lmtc,fisomrel)).T
#     filepath = os.path.join(dataDir / 'simsExp' / 'sensitivity', f'cond_{cond_name}_sens.csv')    
#     pd.DataFrame(data).to_csv(filepath,index=False,header=['Time [s]','Phi [rad]',
#         'STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]','q [ ]','Lmtc [m]','fisomrel [ ]'])

# %% [markdown]
"""
## SSC with different constant knee angular velocities during flexion and extension
"""

#%% SSC with different constant knee angular velocities during flexion and extension
# kje = 1.6 # [rad]
# phi_avg = 1.0 # [rad]

# cf_set  = np.arange(0.4,3.1,0.2) # [rad] n=14
# fts_set = np.arange(0.05,0.96,0.05) # [] n=19
# kje_set = np.arange(0.8,2.05,0.1) # [rad] n=13

# for cf in cf_set: # [Hz]
#     for fts in fts_set:
#         for kje in kje_set: # [rad]
#             print(f'Start SIM for: cf = {cf:0.2f} Hz, fts = {fts:0.2} & kje = {kje:0.2f} rad')
            
#             AMPO, y = derive_predictions.optimise_stim(1,cf,fts,kje,phi_avg,None,muspar,trajectories.cv)
#             time, phi, stim, gamma, lcerel, q, lmtc, lsee, lpee, fisomrel, fsee, fpee, fce, fcerel, vcerel = y
            
#             # Save data
#             data = np.vstack((time,phi,stim,fsee,gamma,lcerel,q,lmtc,fisomrel)).T
#             filepath = os.path.join(dataDir / 'dataSSC', f'cf{cf:{"0.2f"}}Hz_fts{fts:{"0.2f"}}_kje{kje:{"1.2f"}}.csv')
#             pd.DataFrame(data).to_csv(filepath,index=False,header=['Time [s]','Phi [rad]',
#                 'STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]','q [ ]','Lmtc [m]','fisomrel [ ]'])


