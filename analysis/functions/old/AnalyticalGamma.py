# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:59:14 2024

@author: Edwin
"""

import numpy as np

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def AnlyGamma(time, cf, tStimOn, tStimOff, stim, parms):
    """
    AnlyGamma calculates gamma over time for constant muscle stimulation given 
    the cycle frequency, muscle stimulation onset, and offset for fully periodic behavior.

    Inputs:
        time     : The time axis (numpy array)
        cf       : Cycle frequency in Hz
        tStimOn  : Time at which muscle stimulation switches to the constant value of stim
        tStimOff : Time at which muscle stimulation switches to 0
        stim     : Value of constant muscle stimulation
        parms    : A dictionary containing parameters 'gamma_0', 'tact', and 'tdeact'

    Outputs:
        gamma    : Gamma as a function of the input time (numpy array)
        stim     : Stimulation as a function of time (numpy array)
    """
    # Read out parameters
    gamma_0 = parms['gamma_0']  # Minimum value of gamma
    tact = parms['tact']        # Time constant for activation
    tdeact = parms['tdeact']    # Time constant for deactivation

    # Calculate cycle duration
    Tcycle = 1 / cf

    # Shift time with tStimOn and take modulus
    time = np.mod(time - tStimOn, Tcycle)
    tStimOff = np.mod(tStimOff - tStimOn, Tcycle)
    tStimOn = 0

    # Calculate gamma at t=0 for purely periodic behavior
    gamma0_1 = (
        gamma_0 + 
        np.exp(-Tcycle / tdeact) * 
        np.exp(tStimOff / tdeact) * 
        (gamma_0 - 1) * 
        (stim * np.exp(-tStimOff / tact) - stim + 
         (gamma_0 * stim * np.exp(-tStimOff / tact)) / (stim - gamma_0 * stim))
    ) / (
        (stim * np.exp(-Tcycle / tdeact) * 
         np.exp(-tStimOff / tact) * 
         np.exp(tStimOff / tdeact) * 
         (gamma_0 - 1)) / (stim - gamma_0 * stim) + 1
    )

    # Calculate gamma at the end of STIM phase
    gamma0_2 = GammaRise(tStimOff, gamma0_1, stim, parms)

    # Calculate gamma(t)
    gamma = np.where(
        (time >= 0) & (time < tStimOff),
        GammaRise(time, gamma0_1, stim, parms),
        GammaRelax(time - tStimOff, gamma0_2, parms)
    )
    
    # Calculate stim(t)
    stim_t = np.where((time >= 0) & (time < tStimOff), stim, 0)

    return gamma, stim_t

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def GammaRelax(time, gamma0, parms):
    """
    GammaRelax calculates gamma over time given the initial gamma and assuming
    STIM = 0.

    Inputs:
        time   : The time in seconds (numpy array)
        gamma0 : Relative amount of Ca2+ between the filaments at t=0
        parms  : A dictionary containing activation dynamics parameters, 'gamma_0' and 'tdeact'

    Outputs:
        gamma  : Relative amount of Ca2+ between the filaments (numpy array)
    """
    # Unravel parameters
    gamma_0 = parms['gamma_0']
    tdeact = parms['tdeact']

    # Calculations
    # Calculate the time shift which corresponds to gamma(t=0)
    tShift = -np.log((gamma_0 - gamma0) / (gamma_0 - 1)) * tdeact

    # Calculate gamma(t)
    gamma = (1 - gamma_0) * np.exp(-(time + tShift) / tdeact) + gamma_0

    return gamma

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def GammaRise(time, gamma0, stim, parms):
    """
    GammaRise calculates gamma over time given the initial gamma and for a
    constant value of stim.

    Inputs:
        time   : The time in seconds (numpy array)
        gamma0 : Relative amount of Ca2+ between the filaments at t=0
        stim   : Normalized muscle stimulation
        parms  : A dictionary containing activation dynamics parameters, 'gamma_0' and 'tact'

    Outputs:
        gamma  : Relative amount of Ca2+ between the filaments (numpy array)
    """
    # Unravel parameters
    gamma_0 = parms['gamma_0']  # Minimum value of gamma
    tact = parms['tact']        # Inverse of activation time constant

    # Calculations
    # Calculate the time shift which corresponds to gamma(t=0)
    tShift = -np.log((gamma_0 - gamma0) / (stim - gamma_0 * stim) + 1) * tact  # [s]

    # Calculate gamma(t)
    gamma = stim * (1 - gamma_0) * (1 - np.exp(-(time + tShift) / tact)) + gamma_0

    return gamma

