# -*- coding: utf-8 -*-
"""
@author: edhmr, 2021
"""

import numpy as np

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def dino(t, cf, fts, kje, phiAvg, acc, **kwargs):
    """
    Calculates MTC length, velocity, and acceleration at a given point in time for the imposed trajectory.
    A constant acceleration is used around the turning point until a constant shortening/lengthening velocity is reached.


    Calculate the joint angle, angular velocity and angular acceleration over 
    time of the motion imposed by the dino as well as the settings that should
    be used on the dino-pc.

    Inputs:
    - t: time-axis [s]
    - cf: cycle frequency [Hz]
    - fts: fraction of the time shortening (here extension!)
    - kje: knee joint excursion [rad]
    - phiAvg: average knee joint angle [rad]
    - acc: maximum (constant) acceleration of dino [rad/s^2]
    - **kwargs: Optional keyword arguments:
        - 'first_contr' (str): Specifies whether the first contraction is 'F' (flexion) or 'E' (extension). Default is 'F'.
        - 'check' (int): If TRUE, generates plots for checking. Default is FALSE.

    Returns:
    - settings (dict): Dictionary of calculated settings.
    - phi (array): Knee angle over time.
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
    phi += phiAvg-amp
    
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def scv(time, cf, fts, amp, lmtcAvg, acc):
    """
    Calculates MTC length, velocity, and acceleration at a given point in time for the imposed trajectory.
    A sine-shaped acceleration is used around the turning point until a constant shortening/lengthening velocity is reached.

    Inputs:
        time     = the point in time
        cf       = cycle frequency
        fts      = fraction of the cycle time spent shortening
        amp      = MTC-amplitude
        lmtcAvg  = the MTC-length around which the motion is centered
        acc      = the maximum MTC acceleration around the turning points

    Outputs:
        lmtc     = MTC length at the given point in time
        vmtc     = MTC velocity at the given point in time
        amtc     = MTC acceleration at the given point in time
    """

    # Calculate shortening and lengthening times
    tShort = fts / cf  # [s] shortening time
    tLeng = (1 - fts) / cf  # [s] lengthening time

    # Check if inputs are feasible
    # if acc * (acc * np.pi**2 * tLeng**2 + 64 * amp - 16 * amp * np.pi**2) < 0 or acc * np.pi**2 * tShort**2 + 64 * amp - 16 * amp * np.pi**2 < 0:
    #     acc = amp*(16 * np.pi**2 - 64)/(np.pi**2 * tLeng**2)
    #     return None, None, None  # Exit early if the condition is not feasible
        
    # Calculate constant shortening and lengthening velocity
    vShort = (np.pi * np.sqrt(acc * (acc * np.pi**2 * tShort**2 + 64 * amp - 16 * amp * np.pi**2)) - acc * tShort * np.pi**2) / (4 * (np.pi**2 - 4))
    vLeng = -(np.pi * np.sqrt(acc * (acc * np.pi**2 * tLeng**2 + 64 * amp - 16 * amp * np.pi**2)) - acc * tLeng * np.pi**2) / (4 * (np.pi**2 - 4))

    # Calculate acceleration times
    tAcc1 = -vShort / (acc / 2)
    tAcc2 = vLeng / (acc / 2)

    # Calculate constant velocity times
    tCon1 = tShort - 2 * tAcc1
    tCon2 = tLeng - 2 * tAcc2
        
    # Check if inputs are feasible
    # if tCon1 < 0 or tCon2 < 0:
    #     return None, None, None  # Exit early if the condition is not feasible
    #     #  tCon2 <0: acc = (np.pi**2*(64*amp-16*amp*np.pi**2))/((16*tLeng**2-np.pi**4*tLeng**2))
    #     # maybe give it 0.1% more because of numerical precision
    
    # Calculate points in time
    tP = np.cumsum([tAcc1, tCon1, tAcc1, tAcc2, tCon2, tAcc2])
    tP1, tP2, tP3, tP4, tP5, tP6 = tP

    # Current time cycle position
    tc = np.mod(time, tP6)

    # Calculate acceleration
    amtc = np.where(tc <= tP1, -acc / 2 * (np.sin(np.pi / 2 + (np.pi * tc) / tP1) + 1),
                    np.where(tc <= tP2, 0,
                             np.where(tc <= tP3, (np.sin((tc - tP2) * np.pi / (tP3 - tP2) - 0.5 * np.pi) + 1) * acc / 2,
                                      np.where(tc <= tP4, (np.sin((tc - tP3) * np.pi / (tP4 - tP3) + 0.5 * np.pi) + 1) * acc / 2,
                                               np.where(tc <= tP5, 0, 
                                                        (np.sin((tc - tP5) * np.pi / (tP6 - tP5) - 0.5 * np.pi) + 1) * -acc / 2)))))

    # Calculate velocity
    vmtc = np.where(tc <= tP1, -acc / 2 * tc + (-acc / 2 * tP1 * np.sin(np.pi * tc / tP1)) / np.pi,
                    np.where(tc <= tP2, vShort,
                             np.where(tc <= tP3, acc / 2 * (tc - tP3) - (acc / 2 * np.sin(np.pi * (tP2 - tc) / (tP2 - tP3) - np.pi) * (tP2 - tP3)) / np.pi,
                                      np.where(tc <= tP4, acc / 2 * (tc - tP3) + (acc / 2 * np.sin(np.pi * (tP3 - tc) / (tP3 - tP4) - np.pi) * (tP3 - tP4)) / np.pi,
                                               np.where(tc <= tP5, vLeng, 
                                                        -acc / 2 * (tc - tP6) + (-acc / 2 * np.sin(np.pi * (tP5 - tc) / (tP5 - tP6)) * (tP5 - tP6)) / np.pi)))))

    # Initial position conditions
    lmtcP0 = lmtcAvg + amp - ((-acc / 2 * 0**2) / 2 - (-acc / 2 * tP1**2 * np.cos((np.pi * 0) / tP1)) / np.pi**2)
    lmtcP1 = lmtcP0 + (-acc / 2 * tP1**2) / 2 - (-acc / 2 * tP1**2 * np.cos((np.pi * tP1) / tP1)) / np.pi**2
    lmtcP2 = lmtcP1 + (tP2 - tP1) * vShort - ((acc / 2 * tP2**2) / 2 - acc / 2 * tP3 * tP2 + (acc / 2 * np.cos((np.pi * (tP2 - tP2)) / (tP2 - tP3)) * (tP2 - tP3)**2) / np.pi**2)
    lmtcP3 = lmtcP2 + ((acc / 2 * tP3**2) / 2 - acc / 2 * tP3 * tP3 + (acc / 2 * np.cos((np.pi * (tP2 - tP3)) / (tP2 - tP3)) * (tP2 - tP3)**2) / np.pi**2) - ((acc / 2 * (tP3 - tP3)**2) / 2 - (acc / 2 * np.cos((np.pi * (tP3 - tP3)) / (tP3 - tP4)) * (tP3 - tP4)**2) / np.pi**2)
    lmtcP4 = lmtcP3 + ((acc / 2 * (tP3 - tP4)**2) / 2 - (acc / 2 * np.cos((np.pi * (tP3 - tP4)) / (tP3 - tP4)) * (tP3 - tP4)**2) / np.pi**2)
    lmtcP5 = lmtcP4 + (tP5 - tP4) * vLeng - (acc / 2 * tP6 * tP5 - (acc / 2 * tP5**2) / 2 - (acc / 2 * np.cos((np.pi * (tP5 - tP5)) / (tP5 - tP6)) * (tP5 - tP6)**2) / np.pi**2)

    # Calculate position
    lmtc = np.where(tc <= tP1, lmtcP0 + (-acc / 2 * tc**2) / 2 - (-acc / 2 * tP1**2 * np.cos((np.pi * tc) / tP1)) / np.pi**2,
                    np.where(tc <= tP2, lmtcP1 + (tc - tP1) * vShort,
                             np.where(tc <= tP3, lmtcP2 + ((acc / 2 * tc**2) / 2 - acc / 2 * tP3 * tc + (acc / 2 * np.cos((np.pi * (tP2 - tc)) / (tP2 - tP3)) * (tP2 - tP3)**2) / np.pi**2),
                                      np.where(tc <= tP4, lmtcP3 + ((acc / 2 * (tP3 - tc)**2) / 2 - (acc / 2 * np.cos((np.pi * (tP3 - tc)) / (tP3 - tP4)) * (tP3 - tP4)**2) / np.pi**2),
                                               np.where(tc <= tP5, lmtcP4 + (tc - tP4) * vLeng,
                                                        lmtcP5 + (acc / 2 * tP6 * tc - (acc / 2 * tc**2) / 2 - (acc / 2 * np.cos((np.pi * (tP5 - tc)) / (tP5 - tP6)) * (tP5 - tP6)**2) / np.pi**2))))))

    return lmtc, vmtc, amtc
