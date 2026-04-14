# %% [markdown]
# ---
# title: Measurements against predictions
# format:
#   html:
#       css: styles.css
# ---

# %% [markdown]
"""
This script computes the results of the section 'Measured AMPO closely matched 
predicted maximally attainable AMPO'. 

For this, it computes:
    
-   Percent difference between the highest and second highest AMPO within one condition
-   Percent difference between the maximum AMPO of repeated conditions
-   Measured AMPO per participant and condition
-   Standardized z-scores of predicted and measured AMPO
-   Linear regression between predicted and measured AMPO (both normalised and z-scored)
-   Squared Pearson correlation coefficient (r²) for both normalised and z-scored AMPO
    
Custom functions used:
    
-   `kinetics.compute_work(phi, torque, phase, part, iCycles)`
    : Computes net, positive, and negative mechanical work per cycle or per phase
-   `stats.pdiff(value1, value2)`
    : Computes percent difference between two values
-   `stats.zscore(array, axis)`
    : Standardizes array to z-score along the specified axis
-   `kinetics.compute_work(phi, torque, phase, part, iCycles)`
    : Computes net, positive, and negative mechanical work per cycle or per phase
"""

# %% [markdown]
"""
## Load packages & set directories
"""

#%% Load packages
import os, sys
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import kinetics, stats

# %% [markdown]
"""
## Compute percent difference between the highest and second highest AMPO within one condition
"""

#%% Highest vs. second highest AMPO
p_max_vs_2nd = []
ampoDiff = np.nan*np.zeros((6,9))
for iCond, cond in enumerate(['0.40', '0.35C', '0.30', '0.25', '0.20', '0.35A', '0.35B', '0.35D', '0.35E']):
    for iPP,pp in enumerate([1,2,3,4,5,6]):
        filepath = os.path.join(dataDir,'dataExp',f'pp{pp:02d}','AMPO measurements',f'pp{pp:02d}_{cond}_t1.csv')
        df = pd.read_csv(filepath)
        data = df.T.to_numpy()
        
        time,phi,_,torqueComp,_,_,emgQuad,emgHams,phase = data
        
        iCycles = [5,6,7,8,9]
        if pp == 3 and cond == '0.35E':
            iCycles = [6,7,8,9,10] # for this condition pp3 started one cycle too late!
        Wnet,Wpos,Wneg = kinetics.compute_work(phi,torqueComp,phase,'c',iCycles)
               
        WnetSort = np.sort(Wnet)
        p_max_vs_2nd.append(stats.pdiff(WnetSort[-2],WnetSort[-1]))
        
p_max_vs_2nd = np.array(p_max_vs_2nd) # [%] difference between max. and 2nd max
p_max_vs_2nd_mean = f"{p_max_vs_2nd.mean():0.1f}"  # [%] 
p_max_vs_2nd_std  = f"{p_max_vs_2nd.std():0.1f}"  # [%] 

print("Difference of highest vs. second highest AMPO within one condition is: " +
      f"{p_max_vs_2nd_mean} ± {p_max_vs_2nd_std} %")

# %% [markdown]
"""
## Compute percent difference between the maximum AMPO of repeated conditions
"""

#%% Difference between max AMPO of regular and repeated condition
repConds = np.array([['0.40', '0.35C'],
                    ['0.35C', '0.30'],
                    ['0.30', '0.25'],
                    ['0.25', '0.20'],
                    ['0.20', '0.35A'],
                    ['0.35A', '0.35B']])

p_rep = []
iCycles = [5,6,7,8,9]
for iPP,pp in enumerate([1,2,3,4,5,6]):
    for iRep in [0,1]:
        repCond = repConds[iPP,iRep]
        
        filepath = os.path.join(dataDir,'dataExp',f'pp{pp:02d}','AMPO measurements',f'pp{pp:02d}_{repCond}_t1.csv')
        df = pd.read_csv(filepath)
        data = df.T.to_numpy()
        time,phi,_,torqueComp,_,_,emgQuad,emgHams,phase = data
        Wnet = kinetics.compute_work(phi,torqueComp,phase,'c',iCycles)[0]
        
        filepath = os.path.join(dataDir,'dataExp',f'pp{pp:02d}','AMPO measurements',f'pp{pp:02d}_{repCond}_t2.csv')
        df = pd.read_csv(filepath)
        data = df.T.to_numpy()
        time,phi,_,torqueComp,_,_,emgQuad,emgHams,phase = data
        Wnet_rep = kinetics.compute_work(phi,torqueComp,phase,'c',iCycles)[0]  
    
        p_rep.append(stats.pdiff(Wnet_rep.max(),Wnet.max()))

p_rep = np.array(p_rep) # [%] difference between max. and 2nd max
p_rep_mean = f"{p_rep.mean():0.1f}"  # [%] 
p_rep_std  = f"{p_rep.std():0.1f}"  # [%] 

print("Difference in highest AMPO of repeated conditions is: " +
      f"{p_rep_mean} ± {p_rep_std} %")

# %% [markdown]
"""
## Compare measured AMPO against predicted maximally attainable AMPO
"""

# %% Measured vs. predicted AMPO
# Load conditions
data = pd.read_csv(dataDir / 'conditions.csv', sep=',', header=None, dtype=str).to_numpy()

# Set conditions order
cond_names = ['0.20', '0.25', '0.30', '0.35C', '0.40', '0.35A', '0.35B', '0.35D', '0.35E']

# Previously I had a different order, so we need to re-order first
current_order = data[0, :]  # e.g., ['0.25', '0.20', '0.35C', ...]
# Find indices that would reorder to match condNames
reorder_idx = [np.where(current_order == c)[0][0] for c in cond_names]
# Reorder all columns
data = data[:, reorder_idx]

c_cf = data[1].astype(float)
c_fts = data[2].astype(float)
c_ampo = data[3].astype(float)

# Participant info
day, trial = 4, 1
pps = [1, 2, 3, 4, 5, 6]
Npp = len(pps)  # number of participants
filepath = os.path.join(dataDir,'isometric-measurements.csv')
df = pd.read_csv(filepath, index_col=0)

pps = [1,2,3,4,5,6] # participant no.
n_pp = len(pps) # amount of participants
fmax = df['fmax [N]'].to_numpy() # [N] est maximal isometric CE force per particiapnt

# Predictions
ampo_pred = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.35, 0.35, 0.35, 0.35])
ampo_pred = np.repeat(ampo_pred[None, :], Npp, axis=0)

# Compute measured AMPO
ampo_exp = np.full((Npp, len(cond_names)), np.nan)
for i_pp, pp in enumerate(range(1, Npp + 1)):
    for i_cond, cond in enumerate(cond_names):
        data_folder = dataDir / "dataExp" / f"pp{pp:02d}" / "AMPO measurements"
        filename = f"pp{pp:02d}_{cond}_t{trial}"

        df = pd.read_csv(data_folder / f"{filename}.csv")
        data = df.T.to_numpy()
        _, phi, _, torque_comp, *_ , phase = data[0:9]

        # Default cycles
        i_cycles = [5, 6, 7, 8, 9]
        if pp == 3 and cond == "0.35E":
            i_cycles = [6, 7, 8, 9, 10]

        # Compute work
        Wnet, Wpos, Wneg = kinetics.compute_work(phi, torque_comp, phase, part="cycle", iCycles=i_cycles)

        # Normalize to participant's max force and convert to AMPO
        ampo_exp[i_pp, i_cond] = np.max(Wnet * c_cf[i_cond]) / (fmax[i_pp] * 0.093)

# Compute mean and SEM across participants
ampo_exp_mean = np.mean(ampo_exp, axis=0)
ampo_exp_sem = np.std(ampo_exp, axis=0, ddof=1) / np.sqrt(Npp)

# Standardize for z-score
ampo_pred_z = stats.zscore(ampo_exp, axis=-1)
ampo_exp_z = stats.zscore(ampo_pred, axis=-1)

# Linear regressions
fit_norm = sp_stats.linregress(np.ravel(ampo_pred), np.ravel(ampo_exp))
fit_zsco = sp_stats.linregress(np.ravel(ampo_pred_z), np.ravel(ampo_exp_z))

r2_norm = f"{fit_norm.rvalue ** 2:0.2f}"
r2_zsco = f"{fit_zsco.rvalue ** 2:0.2f}"

print(f"r² (norm AMPO): {r2_norm}")
print(f"r² (z-scored AMPO): {r2_zsco}")

# Measured vs. predicted - absolute values (used in discussion)
p_pred_vs_exp_mean = f"{100 + stats.pdiff(ampo_exp,ampo_pred).mean():0.0f}"
p_pred_vs_exp_std  = f"{stats.pdiff(ampo_exp,ampo_pred).std():0.0f}"

print(f"Measued AMPO is {p_pred_vs_exp_mean} ± {p_pred_vs_exp_std} % " +
      "of predicted AMPO")

