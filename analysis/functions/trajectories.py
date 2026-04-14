"""
This module provides functions to calculate knee joint trajectories.

Functions
---------
cv(time, cf, fts, kje, phi_avg, acc=np.inf)
    Compute knee angle and angular velocity using a constant velocity during
    flexion and extension.

dino(t, cf, fts, kje, phi_avg, acc, **kwargs)
    Compute knee angle, angular velocity, and acceleration using a constant-
    acceleration around the transition from flexion-to-extension and vice
    versa.

"""

import numpy as np
import matplotlib.pyplot as plt

def cv(time, cf, fts, kje, phi_avg, acc=np.inf):
    """
    Calculate knee joint angle and angular velocity using a constant velocity model.

    This function computes the knee joint trajectory assuming constant shortening 
    and lengthening velocities during the motion.

    Parameters
    ----------
    time : array_like
        Time points [s] at which to compute the trajectory.
    cf : float
        Cycle frequency [Hz].
    fts : float
        Fraction of the cycle spent shortening (extension).
    kje : float
        Knee joint excursion [rad].
    phi_avg : float
        Average knee joint angle [rad].
    acc : float, optional
        Maximum allowed acceleration [rad/s^2] (default is np.inf, not used in this model).

    Returns
    -------
    phi : ndarray
        Knee joint angle at each time point [rad].
    phid : ndarray
        Knee joint angular velocity at each time point [rad/s].
    """
    # Calculate shortening and lengthening times
    t_ext = fts / cf  # [s] shortening time
    t_flx = (1 - fts) / cf  # [s] lengthening time

    # Calculate constant shortening and lengthening velocity
    v_ext = -kje / t_ext
    v_flx = kje / t_flx

    # Calculate knee angle
    phi = np.where(time < t_ext,
                   phi_avg + kje / 2 + v_ext * time,
                   phi_avg - kje / 2 + v_flx * (time - t_ext))

    # Calculate knee angular velocity
    phid = np.where(time < t_ext, v_ext, v_flx)

    return phi, phid

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def dino(t, cf, fts, kje, phi_avg, acc, **kwargs):
    """
    Calculate knee joint angle, angular velocity, and acceleration for a dino-imposed trajectory.

    The function models the joint motion using constant acceleration around turning points 
    until a constant shortening/lengthening velocity is reached. It also provides settings
    suitable for dino-pc control.

    Parameters
    ----------
    t : array_like
        Time points [s].
    cf : float
        Cycle frequency [Hz].
    fts : float
        Fraction of the cycle spent shortening (extension).
    kje : float
        Knee joint excursion [rad].
    phi_avg : float
        Average knee joint angle [rad].
    acc : float
        Maximum allowed acceleration [rad/s^2].
    **kwargs : optional
        Optional keyword arguments:
        - 'first_contr' : {'F', 'E'}, default 'E'
            Indicates whether the first contraction is flexion ('F') or extension ('E').
        - 'check' : bool, default False
            If True, generates plots for verification.

    Returns
    -------
    phi : ndarray
        Knee joint angle over time [rad].
    phid : ndarray
        Knee joint angular velocity over time [rad/s].
    phidd : ndarray
        Knee joint angular acceleration over time [rad/s^2].
    settings : dict
        Dictionary containing calculated dino settings:
        - 'CyclusDuration' : cycle duration [ms]
        - 'StartAngle' : initial angle [deg]
        - 'Displacement' : amplitude [deg]
        - 'Speed' : list of angular velocities [deg/s]
        - 'Acceleration' : list of accelerations [deg/s^2]
        - 'Deceleration' : list of decelerations [deg/s^2]

    Raises
    ------
    ValueError
        If the provided parameters produce an impossible configuration.
    """
    
    # Set default values for 'first_contr' and 'check'
    first_contr = kwargs.get('first_contr', 'E')  # Default is 'E' (extension)
    check = kwargs.get('check', False)  # Default is False (no check)
    
    # First check whether this condition is possible
    maxFTS = 1 - (2*cf*kje**(1/2))/acc**(1/2)
    if abs(maxFTS - 0.5) < abs(fts - 0.5) + 1e-15:  # If FTS > maxFTS, this condition is impossible
        raise ValueError("Condition not possible.")
    
    # Calculate constant angular velocity and other parameters
    tExt = fts / cf  # [s] extension time
    tFlx = (1 - fts) / cf  # [s] flexion time
    
    # Calculate constant flexion & extension angular velocities
    phidExt = -(acc * (tExt - (-(-acc * tExt ** 2 + 4 * kje) / acc) ** 0.5)) / 2  # [rad/s]
    phidFlx = (acc * (tFlx - (-(-acc * tFlx ** 2 + 4 * kje) / acc) ** 0.5)) / 2  # [rad/s]
        
    # Calculate points in time (acceleration, constant, deceleration)        
    tA = [-phidExt / acc, phidFlx / acc]
    tC = [tExt - 2 * tA[0], tFlx - 2 * tA[1]]
    phid = [phidExt, phidFlx]
    tP = [tFlx, tExt]
    acc = [-acc, acc]
    amp = -kje/2
        
    if first_contr in ['F', 'Flx', 'Flex', 'Flexion']:  # If we start with flexion, reverse the order
        tA = tA[::-1]
        tC = tC[::-1]
        phid = phid[::-1]
        tP = tP[::-1]
        acc = acc[::-1]
        amp = -amp
                
    # Calculate the points in time for acceleration, constant velocity, deceleration
    tP = np.cumsum([tA[0], tC[0], tA[0], tA[1], tC[1], tA[1]])

    # Time array
    tc = np.mod(t, tP[5])

    # Calculate the knee angle (phi)
    phi = np.piecewise(
        tc,
        [
            tc <= tP[0],
            (tc > tP[0]) & (tc <= tP[1]),
            (tc > tP[1]) & (tc <= tP[3]),
            (tc > tP[3]) & (tc <= tP[4]),
            (tc > tP[4]) & (tc <= tP[5]),
        ],
        [
            lambda tc: 0.5 * (tc ** 2) * acc[0],
            lambda tc: (tc - tP[0]) * phid[0] + 0.5 * (tP[0] ** 2) * acc[0],
            lambda tc: 0.5 * (tc - tP[2]) ** 2 * acc[1] - 0.5 * (tP[1] - tP[2]) ** 2 * acc[1] + (tP[1] - tP[0]) * phid[0] + 0.5 * (tP[0] ** 2) * acc[0],
            lambda tc: (tc - tP[3]) * phid[1] + 0.5 * (tP[3] - tP[2]) ** 2 * acc[1] - 0.5 * (tP[1] - tP[2]) ** 2 * acc[1] + (tP[1] - tP[0]) * phid[0] + 0.5 * (tP[0] ** 2) * acc[0],
            lambda tc: 0.5 * (tc - tP[5]) ** 2 * acc[0] - 0.5 * (tP[4] - tP[5]) ** 2 * acc[0] + (tP[4] - tP[3]) * phid[1] + 0.5 * (tP[3] - tP[2]) ** 2 * acc[1] - 0.5 * (tP[1] - tP[2]) ** 2 * acc[1] + (tP[1] - tP[0]) * phid[0] + 0.5 * (tP[0] ** 2) * acc[0],
        ],
    )
    phi += phi_avg-amp
    
    # Calculate the knee angular velocity (phid)
    phid = np.piecewise(
        tc,
        [
            tc <= tP[0],
            (tc > tP[0]) & (tc <= tP[1]),
            (tc > tP[1]) & (tc <= tP[3]),
            (tc > tP[3]) & (tc <= tP[4]),
            (tc > tP[4]) & (tc <= tP[5]),
        ],
        [
            lambda tc: tc * acc[0],
            lambda tc: phid[0],
            lambda tc: phid[0] + (tc - tP[1]) * acc[1],
            lambda tc: phid[1],
            lambda tc: phid[1] + (tc - tP[4]) * acc[0],
        ],
    )
    
    # Calculate the knee angular aceleration (phidd)
    phidd = np.piecewise(
        tc,
        [
            tc <= tP[0],
            (tc > tP[0]) & (tc <= tP[1]),
            (tc > tP[1]) & (tc <= tP[3]),
            (tc > tP[3]) & (tc <= tP[4]),
            (tc > tP[4]) & (tc <= tP[5]),
        ],
        [
            lambda tc: acc[0],
            lambda tc: 0,
            lambda tc: acc[1],
            lambda tc: 0,
            lambda tc: acc[0],
        ],
    )

    # Dino settings
    settings = {
        'CyclusDuration': 1 / cf * 1000,  # [ms]
        'StartAngle': float(phi[0]) * 180 / np.pi,  # [deg]
        'Displacement': amp * 180 / np.pi,  # [deg]
        'Speed': [phid[0] * 180 / np.pi, phid[1] * 180 / np.pi] ,  # [deg/s]
        'Acceleration': [acc[0] * 180 / np.pi, acc[1] * 180 / np.pi],  # [deg/s^2]
        'Deceleration': [-acc[0] * 180 / np.pi, -acc[1] * 180 / np.pi],  # [deg/s^2]
    }

    # Plot for checking purposes if 'check' is provided
    if check in [1, True]:
        nCycle = int(np.ceil(t[-1]/tP[5]))
        if first_contr in ['F', 'Flx', 'Flex', 'Flexion']:  # If we start with flexion
            xtickP1 = [0] +  [tFlx+x for x in range(0, nCycle)] + [x/cf for x in range(1, nCycle+1)]
            xtickC1 = ['0'] + [f'({x}-FTS)/CF' for x in range(1, nCycle+1)] + [f'{x}/CF' for x in range(1, nCycle+1)]
            xtickP2 = [0] +  [x-fts for x in range(1, nCycle+1)] + [x for x in range(1, nCycle+1)]
            xtickC2 = ['0'] + [f'({x}-FTS)' for x in range(1, nCycle+1)] + [f'{x}' for x in range(1, nCycle+1)]
        else:
            xtickP1 = [0] + [tExt+x/cf for x in range(0, nCycle)] + [x/cf for x in range(1, nCycle+1)] 
            xtickC1 = ['0'] + [f'({x}+FTS)/CF' for x in range(0, nCycle)] + [f'{x}/CF' for x in range(1, nCycle+1)]
            xtickP2 = [0] + [x+fts for x in range(0, nCycle)] + [x for x in range(1, nCycle+1)]
            xtickC2 = ['0'] + [f'{x}+FTS' for x in range(0, nCycle)] + [f'{x}' for x in range(1, nCycle+1)]
        
        plt.figure(figsize=(10, 8))
        
        # Plot for checking knee angle vs time
        plt.subplot(2, 1, 1)
        plt.plot(t, phi)
        plt.title('Figure for checking purposes..')
        plt.xlabel('Time [s]')
        plt.ylabel('Knee angle [rad]')
        plt.xticks(xtickP1, xtickC1)
        plt.grid(True)

        # Plot normalized time
        plt.subplot(2, 1, 2)
        plt.plot(t / tP[5], phi)
        plt.xlabel('Normalized time')
        plt.ylabel('Knee angle [rad]')
        plt.xticks(xtickP2, xtickC2)
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    
    return phi, phid, phidd, settings