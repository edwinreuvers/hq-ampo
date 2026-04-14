#%% Imports etc.
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import stats

plt.close('all')

#%% Data
parFile = os.path.join(dataDir,'VAS_muspar.pkl')
muspar = pickle.load(open(parFile, 'rb'))

acc = 50 # [rad/s^2]
kje = 1.6 # [rad]
maxcf = (0.5)*acc**(1/2)/2/kje**(1/2) # maximum CF that can be imposed given acc and amp

cf_set = np.arange(np.floor(maxcf*100)/100,0.33,-0.02)[::-1] # range of CF that we can investigate
fts_max = 1 - (2*cf_set*kje**(1/2))/acc**(1/2) # maximum fts that can be imposed given acc, amp and cf
fts_wide_range = np.hstack([1-stats.floor(fts_max,3), stats.floor(fts_max[::-1],3)]) # maximum range of cf

ampo = np.full((len(fts_wide_range),len(cf_set)),np.nan)
for iCf,cf in enumerate(cf_set): # [Hz]
    fts_set = fts_wide_range[iCf:-iCf]
    for iFts,fts in enumerate(fts_set):  
        filepath = os.path.join(dataDir,'simsExp','contour',f'kje{kje:0.2f}rad_cf{cf:0.2f}Hz_fts{fts:0.3f}.csv')
        try:
            df = pd.read_csv(filepath)
            data = df.T.to_numpy()
            time,phi,stim,fsee,gamma,lcerel,q,lmtc,fisomrel = data
        
            ampo[iFts+iCf,iCf] = -np.trapezoid(fsee*muspar['A1'],phi)*cf/(muspar['fmax']*muspar['lce_opt'])
        except:
            print(filepath)
