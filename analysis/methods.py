# %% [markdown]
# ---
# title: Methods
# ---

# %% [markdown]
"""
On this page all variables are computed that are presented in the Methods

Specifically, the following was done:
          
-   Compute the change in AMPO for Lceopt with a 10% decrease Lceopt

"""

# %% [markdown]
"""
## Load packages & set directories
"""

#%%
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import stats

# %% [markdown]
"""
## Compute percentage difference in AMPO
"""

#%% Compute difference in AMPO
# Load conditions
data = pd.read_csv(dataDir / 'conditions.csv', sep=',', header=None, dtype=str).to_numpy()
cond_names = data[0]
cf_set = data[1].astype(float)
fts_set = data[2].astype(float)

# Loop over files
AMPO_org = np.full(len(cond_names), np.nan)
AMPO_sns = np.full(len(cond_names), np.nan)
for i,(cf,fts,cond_name) in enumerate(zip(cf_set,fts_set,cond_names)):
    filepath = os.path.join(dataDir,'simsExp',f'cond_{cond_name}.csv')
    df = pd.read_csv(filepath)
    data = df.to_numpy().T
    time,phi,stim,fsee,gamma,lcerel,q,lmtc,fisomrel = data
    AMPO_org[i] = -np.trapezoid(fsee,lmtc)*cf/(5250*0.093)
    
    filepath = os.path.join(dataDir,'simsExp','sensitivity', f'cond_{cond_name}_sens.csv')
    df = pd.read_csv(filepath)
    data = df.to_numpy().T
    time,phi,stim,fsee,gamma,lcerel,q,lmtc,fisomrel = data
    AMPO_sns[i] = -np.trapezoid(fsee,lmtc)*cf/(5250*0.093)

p_diff = stats.pdiff(np.array(AMPO_sns),np.array(AMPO_org))
p_diff_mean = p_diff.mean()
p_diff_std = p_diff.std()

print("AMPO of 10% decrease AMPO vs. normal Lceopt is: " +
      f"{p_diff_mean:0.1f} ± {p_diff_std:0.1f} %")
