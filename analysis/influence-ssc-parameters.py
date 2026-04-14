# %% [markdown]
# ---
# title: Influence SSC parameters
# ---

# %% [markdown]
"""
This script computes the results of the section 'Influence of SSC parameters 
on the maximally attainable AMPO'. 

Specifically, it performs the following steps:
    
-   Loads muscle parameters from a pickle file (`VAS_muspar.pkl`).
-   Defines parameter grids for SSC parameters: 
    -   CF (cycle frequency)
    -   FTS (fraction of the cycle time spent shortening)
    -   KJE (knee joint excursion)
-   Loads precomputed simulation data for each combination of SSC parameters.
-   Computes the AMPO (average mechanical power output) normalised by 
    maximal isometric CE force (fmax) and optimal CE length (lceopt).
-   Interpolates the AMPO values onto a finer 3D grid for smoother analysis.
-   Determines the maximum AMPO and the corresponding optimal SSC parameters.

"""

# %% [markdown]
"""
## Load packages & set directories
"""

#%% Imports etc.
import os, sys, pickle
import numpy as np
import pandas as pd
from scipy import integrate, interpolate
from pathlib import Path

# Paths
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

# %% [markdown]
"""
## Extract AMPO of each condition
"""

#%% Extract AMPO of each condition
# Load Parameters
parFile = os.path.join(dataDir, 'VAS_muspar.pkl')
muspar = pickle.load(open(parFile, 'rb'))

# Define parameter grids
kjeSet = np.arange(0.8, 2.01, 0.1)
cfSet = np.arange(0.4, 3.1, 0.2)
ftsSet = np.arange(0.05, 0.96, 0.05)

# Initialize AMPO array
AMPO = np.full((len(kjeSet), len(ftsSet), len(cfSet)), np.nan)

# Load simulation data and compute AMPO
for iKJE, kje in enumerate(kjeSet):
    for iFTS, fts in enumerate(ftsSet):
        for iCF, cf in enumerate(cfSet):
            filepath = os.path.join(dataDir, 'simsSSC',
                                    f'cf{cf:0.2f}Hz_fts{fts:0.2f}_kje{kje:0.2f}rad.csv')
            try:
                df = pd.read_csv(filepath)
                data = df.T.to_numpy()
                time, phi, stim, fsee, gamma, lcerel, q, lmtc, fisomrel = data
                # Compute AMPO normalized by fmax and lce_opt
                AMPO[iKJE, iFTS, iCF] = -integrate.trapezoid(fsee * muspar['A1'], phi) * cf / (muspar['fmax'] * muspar['lce_opt'])
            except Exception:
                AMPO[iKJE, iFTS, iCF] = np.nan

# Create meshgrid for axes (match AMPO shape)
Z, Y, X = np.meshgrid(kjeSet, ftsSet, cfSet, indexing='ij')  # ij indexing: Z=kje, Y=fts, X=cf

# Flatten data for interpolation
x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = Z.flatten()
data_flat = AMPO.flatten()

# Define finer interpolation grid
kje_fine = np.linspace(kjeSet[0], kjeSet[-1], 100)
fts_fine = np.linspace(ftsSet[0], ftsSet[-1], 100)
cf_fine  = np.linspace(cfSet[0], cfSet[-1], 100)

Zf, Yf, Xf = np.meshgrid(kje_fine, fts_fine, cf_fine, indexing='ij')

# Interpolate
AMPO_fine = interpolate.griddata((z_flat, y_flat, x_flat), data_flat, (Zf, Yf, Xf), method='linear')

# Find maximum AMPO and corresponding parameters
max_idx = np.nanargmax(AMPO_fine)
iKJE, iFTS, iCF = np.unravel_index(max_idx, AMPO_fine.shape)

opt_kje  = f"{Zf[iKJE, iFTS, iCF]:0.1f}"
opt_fts  = f"{Yf[iKJE, iFTS, iCF]:0.2f}"
opt_cf   = f"{Xf[iKJE, iFTS, iCF]:0.1f}"
maxAMPO = AMPO_fine[iKJE, iFTS, iCF]

# Print results
print(f"Maximum AMPO: {maxAMPO:0.3f}")
print(f"Optimal KJE: {opt_kje} rad, FTS: {opt_fts}, CF: {opt_cf} Hz")