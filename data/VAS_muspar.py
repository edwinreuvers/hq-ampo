# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:16:15 2024

@author: Edwin
"""

import pickle, os
import numpy as np

cwd = os.getcwd()
baseDir = os.path.join(cwd,'..')
dataDir = os.path.join(baseDir,'data')

muspar = {}

#%% General
# Activation dynamics
muspar['q0'] = 0.005
muspar['tact'] = 1/11.25
muspar['tdeact'] = 1/11.25
muspar['a_act'] = -4.587
muspar['b_act'] = np.array([5.1680, 1.0810, -0.1909])
muspar['gamma_0'] = 1e-5
muspar['kCa'] = 8e-6

# Force-length relationship
muspar['w'] = 0.56
muspar['n'] = 2
muspar['lce_opt'] = 0.093 # [m]
muspar['fmax'] = 5250 # [N]
muspar['lsee0'] = 0.160 # [m]
muspar['ksee'] = muspar['fmax']/(0.04*muspar['lsee0'])**2

lpee0 = 1.4*muspar['lce_opt']
lpee1 = 1.56 * muspar['lce_opt']     # where fpee = 0.5 * fmax
fpee1 = 0.5 * muspar['fmax']
kpee = fpee1 / (lpee1 - lpee0)**2
muspar['lpee0'],muspar['kpee'] = lpee0, kpee

# Force-velocity relationship
muspar['arel'] = 0.41
muspar['brel'] = 5.2
muspar['a'] = 0.41*muspar['fmax']
muspar['b'] = 5.2*muspar['lce_opt']
muspar['fasymp'] = 1.5
muspar['slopfac'] = 2
muspar['vfactmin'] = 0.1

# Force-velocity relationship
muspar['A0'] = 0.213053
muspar['A1'] = 0.042

fileName = os.path.join(dataDir,'VAS_muspar.pkl')
pickle.dump(muspar, open(fileName, 'wb'))