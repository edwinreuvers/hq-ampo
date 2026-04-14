# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:02:46 2024

@author: Edwin
"""

import numpy as np
from FuncGen import data2List

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_work(phi,tor,phase,part='cycle',iCycles = None):
    """
    Compute mechanical work (net, positive, negative) from torque-angle loops.

    Parameters
    ----------
    phi : list of np.ndarray
        Knee angle per cycle
     : list of np.ndarray
        Torque per cycle
    phase : list of np.ndarray
        (kept for backward compatibility)
    part : str
        'cycle', 'flexion', or 'extension'
    iCycles : list-like, optional
        Indices of cycles to include

    Returns
    -------
    Wnet, Wpos, Wneg : np.ndarray
        Net, positive, and negative work
    """

    
    #%% Find start flexion and extension phases
    i_sf = [idx for idx,item in enumerate(phase) if "sf" in item]
    i_se = [idx for idx,item in enumerate(phase) if "se" in item]
    
    #%% Extract knee angle and torque during the whole cycle, 
    # and the flexion and extension phase   
    if iCycles == None:
        iCycles = np.arange(0,len(i_se)-1)

    phi_cyc     = data2List(phi,phase,iCycles,'c')
    tor_cyc     = data2List(tor,phase,iCycles,'c')
    phi_flx      = data2List(phi,phase,iCycles,'f')
    tor_flx     = data2List(tor,phase,iCycles,'f')
    phi_ext     = data2List(phi,phase,iCycles,'e')
    tor_ext     = data2List(tor,phase,iCycles,'e')
    
    #%% Now calculate MW per cycle etc.
    Ncycli = len(iCycles)
    Wnet_cyc = np.nan*np.zeros(Ncycli)
    Wnet_flx = np.nan*np.zeros(Ncycli)
    Wnet_ext = np.nan*np.zeros(Ncycli)
    Wneg_cyc = np.nan*np.zeros(Ncycli) 
    Wneg_flx = np.nan*np.zeros(Ncycli)
    Wneg_ext = np.nan*np.zeros(Ncycli) 
    Wpos_flx = np.nan*np.zeros(Ncycli)
    Wpos_ext = np.nan*np.zeros(Ncycli)
    
    iCycles = [*map(lambda x: x - iCycles[0], iCycles)]
    for idx in iCycles:
        phi = phi_cyc[idx]
        tor = tor_cyc[idx]
        Wnet_cyc[idx] = -np.trapz(tor,phi)
        
        # Then during the flexion
        phi = phi_flx[idx]
        tor = tor_flx[idx]
        Wnet_flx[idx] = -np.trapz(tor,phi)        
        Wneg_flx[idx] = -np.trapz(tor[tor>0],phi[tor>0])
        Wpos_flx[idx] = -np.trapz(tor[tor<0],phi[tor<0])
        
        # And extension
        phi = phi_ext[idx]
        tor = tor_ext[idx]
        Wnet_ext[idx] = -np.trapz(tor,phi)
        Wneg_ext[idx] = -np.trapz(tor[tor<0],phi[tor<0])
        Wpos_ext[idx] = -np.trapz(tor[tor>0],phi[tor>0])
    
    Wpos_cyc = Wpos_ext+Wpos_flx
    Wneg_cyc = Wneg_ext+Wneg_flx
    
    #%%
    if part in ['cycle', 'cyc', 'c']:
        return Wnet_cyc, Wpos_cyc, Wneg_cyc
    elif part in ['flexion', 'flex', 'flx', 'f']:
        return Wnet_flx, Wpos_flx, Wnet_flx
    elif part in ['extension', 'ext', 'e']:
        return Wnet_ext,  Wpos_ext, Wneg_ext
    else:
        return None, None, None