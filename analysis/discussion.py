# %% [markdown]
# ---
# title: Discussion
# ---

# %% [markdown]
"""
On this page all variables are computed that are presented in the Discussion

Specifically, the following was done:
          
-   Comparison relation knee joint angular velocity-knee joint moment 
    relation with Chen & Franklin (2025)
-   Compute decrease in AMPO from FTS 0.8 to 0.5

Custom functions used:

-   hillmodel.fce2vce(fce,q,lcerel,muspar)
    : Computes CE shortening velocity for given CE force, active state and CE length
-   stats.pdiff(value1, value2)
    : Computes percent difference between two values
    
"""

# %% [markdown]
"""
## Load packages & set directories
"""

#%%
import os, sys, pickle
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path

# Paths
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

plt.close('all')

import hillmodel, stats

# %% [markdown]
"""
## Calibrate figure of Chen & Franklin (2025)
"""

#%% Calibrate figure of Chen & Franklin (2025)
# Load the PNG
figuredata = imread('moment_vs_velocity.png')
f, ax = plt.subplots(figsize=(17.2, 16))
ax.imshow(figuredata)

# --- Step 3: Define mapping ---
# Known values from the axis ticks
xlab = [-360, 0, 420]
ylab = [-150, 0, 150]

# Extract pixel calibration
xcal = np.array([[74.20806916286995, 111.42320067102327],
                 [441.99742399780393, 111.64231896670498],
                 [872.5227001549637, 111.49017741864587]])
ycal = np.array([[904.5312301797321, 685.3215087341092],
                 [903.8586797684089, 398.9368072980768],
                 [904.5213618220112, 111.54829302145951]])


plt.plot(xcal[:,0],xcal[:,1],'ro')
plt.plot(ycal[:,0],ycal[:,1],'go')

# Calibration
xcoeff = np.polyfit(xcal[:, 0], xlab, 1)
ycoeff = np.polyfit(ycal[:, 1], ylab, 1)

# %% [markdown]
"""
## Show calibrate figure with model prediction
"""

#%% Show calibrate figure with model prediction
# # Create the scaled vectors for the image
imageplotvectorx = np.polyval(xcoeff, [0, figuredata.shape[1]])
imageplotvectory = np.polyval(ycoeff, [0, figuredata.shape[0]])

# Plot calibrated figure
f, ax = plt.subplots(figsize=(17.2, 16))
ax.imshow(figuredata, extent=[imageplotvectorx[0], imageplotvectorx[1], imageplotvectory[1], imageplotvectory[0]], aspect='auto')
ax.set_title('Calibrated')
ax.grid(True)

# Load Parameters
parFile = os.path.join(dataDir,'VAS_muspar.pkl')
muspar = pickle.load(open(parFile, 'rb'))

# Compute model relationship
fce = np.linspace(0,1,100)*muspar['fmax']
q = np.ones_like(fce)
lcerel = np.ones_like(fce)
vce = hillmodel.fce2vce(fce,q,lcerel,muspar)[0]

M = fce*muspar['A1']
Mnorm = M/M.max()
phidot = vce/muspar['A1']/np.pi*180

plt.plot(-phidot,-Mnorm*100,'r')

# %% [markdown]
"""
## Compute decrease in AMPO from FTS 0.8 to 0.5
"""

#%% Compute decrease in AMPO from FTS 0.8 to 0.5
# Load Parameters
parFile = os.path.join(baseDir,'predictions','VAS_muspar.pkl')
muspar = pickle.load(open(parFile, 'rb'))

# Grid
kjeSet = np.arange(0.8,2.01,0.4) 
cfSet = np.arange(0.4,3.1,0.2)  
ftsSet = np.arange(0.5,0.96,0.05)

p_fts_08_to_05 = np.full((len(kjeSet),100),np.nan)
p_fts_08_to_05_opt = np.full(len(kjeSet),np.nan)
for iKje,kje in enumerate(kjeSet):
    AMPO = np.empty((len(ftsSet),len(cfSet)))
    for iCf,cf in enumerate(cfSet):
        for iFts,fts in enumerate(ftsSet):
            filepath = os.path.join(dataDir,'simsSSC',f'cf{cf:{"0.2f"}}Hz_fts{fts:{"0.2f"}}_kje{kje:{"0.2f"}}rad.csv')   
            df = pd.read_csv(filepath)
            data = df.T.to_numpy()
            time,phi,stim,fsee,gamma,lcerel,q,lmtc,fisomrel = data
            
            AMPO[iFts,iCf] = -np.trapezoid(fsee*muspar['A1'],phi)*cf/(muspar['fmax']*muspar['lce_opt'])
    
    X, Y = np.meshgrid(cfSet,ftsSet)
    
    # Finer mesh
    data = AMPO

    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = data.flatten()

    # Remove NaN values for interpolation
    mask = ~np.isnan(z_flat)
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]
    z_flat = z_flat[mask]

    # Define a finer grid for interpolation
    x_fine = np.linspace(cfSet[0],cfSet[-1],100); x_fine[-1] = cfSet[-1]
    y_fine = np.linspace(ftsSet[0],ftsSet[-1],200); y_fine[-1] = ftsSet[-1]
    x_fine, y_fine = np.meshgrid(x_fine, y_fine)

    # Perform the interpolation
    z_fine = griddata((x_flat, y_flat), z_flat, (x_fine, y_fine), method='cubic')
    
    # Find maximum
    max_idx = np.nanargmax(z_fine)
    iRow, iCol = np.unravel_index(max_idx, z_fine.shape)
    
    # Difference from FTS 0.5 to 0.8
    p_fts_08_to_05[iKje] = stats.pdiff(z_fine[iRow,:],z_fine[0,:]) # [%] @ any CF
    p_fts_08_to_05_opt[iKje] = stats.pdiff(z_fine[iRow,iCol],z_fine[0,iCol]) # [%] @ opt CF
    
# Range of improvement going FTS 0.5 -> 0.8 (round to nearest 5%)
p_fts_08_to_05_min = np.min(np.round(p_fts_08_to_05_opt*4,-1)/4) # [%]
p_fts_08_to_05_max = np.max(np.round(p_fts_08_to_05_opt*4,-1)/4) # [%]

