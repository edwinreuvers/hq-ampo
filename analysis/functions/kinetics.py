"""
This module provides functions to compute net, positive, and negative
mechanical work from joint angle and moment signals over repeated cycles.
It relies on phase labels (e.g., 'sf' for start flexion and 'se' for start
extension) to segment the data into meaningful biomechanical phases.

The main functionality includes:
- Segmenting signals into cycles, flexion, and extension phases.
- Computing mechanical work using numerical integration (trapezoidal rule).
- Separating positive and negative work contributions.

Functions
---------
compute_work(phi, mom, phase, part='cycle', iCycles=None)
    Compute net, positive, and negative mechanical work for specified
    portions of the movement cycle (full cycle, flexion, or extension).
"""

import numpy as np
import helpers

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_work(phi,mom,phase,part='cycle',iCycles=None):
    """
    Compute mechanical work per cycle.

    Parameters
    ----------
    phi : list of np.ndarray
        Knee angle per cycle
    mom : list of np.ndarray
        Moment per cycle
    phase : ignored or optional
        (kept for backward compatibility)
    part : str
        'cycle', 'flexion', or 'extension'
    iCycles : list-like, optional
        Indices of cycles to include

    Returns
    -------
    Wnet, Wpos, Wneg : np.ndarray
    """
    
    # Find start flexion and extension phases
    i_sf = [idx for idx,item in enumerate(phase) if "sf" in item]
    i_se = [idx for idx,item in enumerate(phase) if "se" in item]
    
    # Extract knee angle and moment during the whole cycle, 
    # and the flexion and extension phase   
    if iCycles == None:
        iCycles = np.arange(0,len(i_se)-1)

    phi_cyc     = helpers.segments_to_list(phi,phase,'c',iCycles)
    mom_cyc     = helpers.segments_to_list(mom,phase,'c',iCycles)
    phi_flx     = helpers.segments_to_list(phi,phase,'f',iCycles)
    mom_flx     = helpers.segments_to_list(mom,phase,'f',iCycles)
    phi_ext     = helpers.segments_to_list(phi,phase,'e',iCycles)
    mom_ext     = helpers.segments_to_list(mom,phase,'e',iCycles)
    
    #%% Now calculate MW per cycle etc.
    n = len(iCycles)
    Wnet_cyc = np.full(n, np.nan)
    Wnet_flx = np.full(n, np.nan)
    Wnet_ext = np.full(n, np.nan)
    Wneg_flx = np.full(n, np.nan)
    Wpos_flx = np.full(n, np.nan)
    Wneg_ext = np.full(n, np.nan) 
    Wpos_ext = np.full(n, np.nan)
    
    iCycles = [*map(lambda x: x - iCycles[0], iCycles)]
    for idx in iCycles:
        phi = phi_cyc[idx]
        mom = mom_cyc[idx]
        Wnet_cyc[idx] = -np.trapz(mom,phi)
        
        # Then during the flexion
        phi = phi_flx[idx]
        mom = mom_flx[idx]
        Wnet_flx[idx] = -np.trapz(mom,phi)        
        Wneg_flx[idx] = -np.trapz(mom[mom>0],phi[mom>0])
        Wpos_flx[idx] = -np.trapz(mom[mom<0],phi[mom<0])
        
        # And extension
        phi = phi_ext[idx]
        mom = mom_ext[idx]
        Wnet_ext[idx] = -np.trapz(mom,phi)
        Wneg_ext[idx] = -np.trapz(mom[mom<0],phi[mom<0])
        Wpos_ext[idx] = -np.trapz(mom[mom>0],phi[mom>0])
    
    Wpos_cyc = Wpos_ext+Wpos_flx
    Wneg_cyc = Wneg_ext+Wneg_flx
    
    # Return
    part = part.lower()
    if part in ['cycle', 'cyc', 'c']:
        return Wnet_cyc, Wpos_cyc, Wneg_cyc
    elif part in ['flexion', 'flex', 'flx', 'f']:
        return Wnet_flx, Wpos_flx, Wneg_flx
    elif part in ['extension', 'ext', 'e']:
        return Wnet_ext,  Wpos_ext, Wneg_ext
    else:
        return None, None, None