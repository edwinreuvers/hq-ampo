# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 12:12:18 2025

@author: Edwin
"""

import numpy as np
import pandas as pd

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_ampo(filepaths):
    # Case 0: Single string input
    if isinstance(filepaths, str):
        try:
            data = pd.read_csv(filepaths).T.to_numpy()
            time, phi, _, fsee, _,_,_,lmtc,*_ = data
            return -np.trapz(fsee, lmtc) / time[-1]
        except Exception:
            return np.nan
    # Convert to NumPy array to check dimensions
    filepaths_arr = np.array(filepaths, dtype=object)

    # Case 12: 1D list of filepaths
    if filepaths_arr.ndim == 1:
        AMPOs = []
        for filepath in filepaths_arr:
            try:
                data = pd.read_csv(filepath).T.to_numpy()
                time, phi, _, fsee, _,_,_,lmtc,*_ = data
                AMPO = -np.trapz(fsee, lmtc) / time[-1]
                AMPOs.append(AMPO)
            except Exception:
                AMPOs.append(np.nan)
        return np.array(AMPOs)

    # Case 2: 2D list of filepaths
    elif filepaths_arr.ndim == 2:
        shape = filepaths_arr.shape
        AMPOs = np.full(shape, np.nan, dtype=float)
        for i in range(shape[0]):
            for j in range(shape[1]):
                try:
                    data = pd.read_csv(filepaths_arr[i, j]).T.to_numpy()
                    time, phi, _, fsee, _,_,_,lmtc,*_ = data
                    AMPO = -np.trapz(fsee, lmtc) / time[-1]
                    AMPOs[i, j] = AMPO
                except Exception:
                    continue  # Already NaN
        return AMPOs
    
    # Case 3: 3D list of filepaths
    elif filepaths_arr.ndim == 3:
        shape = filepaths_arr.shape
        AMPOs = np.full(shape, np.nan, dtype=float)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    try:
                        data = pd.read_csv(filepaths_arr[i, j, k]).T.to_numpy()
                        time, phi, _, fsee, _,_,_,lmtc,*_ = data
                        AMPO = -np.trapz(fsee, lmtc) / time[-1]
                        AMPOs[i, j, k] = AMPO
                    except Exception:
                        continue  # Already NaN
        return AMPOs
    else:
        raise ValueError("Unsupported input structure for 'filepaths'.")