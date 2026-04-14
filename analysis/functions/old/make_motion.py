# -*- coding: utf-8 -*-
"""
@author: edhmr, 2021
"""

import numpy as np

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def dino(time, cf, fts, amp, phiAvg, acc):
    """
    Calculates MTC length, velocity, and acceleration at a given point in time for the imposed trajectory.
    A constant acceleration is used around the turning point until a constant shortening/lengthening velocity is reached.

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
    tExt = fts / cf  # [s] shortening time
    tFlx = (1 - fts) / cf  # [s] lengthening time
    
    # Check if inputs are feasible
    maxFTS = 1 - (2*cf*(amp*2)**(1/2))/acc**(1/2)
    if abs(maxFTS-0.5) < abs(fts-0.5)+1e-15: # if FTS > maxFTS dan kan deze conditie niet..
        return None, None, None
 
    # Calculate constant shortening and lengthening velocity
    vExt = -(acc*(tExt - (-(- acc*tExt**2 + 8*amp)/acc)**(1/2)))/2  # [rad/s]
    vFlx = (acc*(tFlx - (-(- acc*tFlx**2 + 8*amp)/acc)**(1/2)))/2 # [rad/s]
    
    # Calculate acceleration times
    tAcc1 = -vExt/acc
    tAcc2 = vFlx/acc

    # Calculate constant velocity times
    tCon1 = tExt - 2 * tAcc1
    tCon2 = tFlx - 2 * tAcc2
     
    # Accelerations
    acc1 = -acc
    acc2 = acc
    
    
    # Check if inputs are feasible
    if tCon1 < 0 or tCon2 < 0:
        return None, None, None  # Exit early if the condition is not feasible

    # Calculate points in time
    tP = np.cumsum([tAcc1, tCon1, tAcc1, tAcc2, tCon2, tAcc2])
    tP1, tP2, tP3, tP4, tP5, tP6 = tP

    # Current time cycle position
    tc = np.mod(time, tP6)
    
    phi = np.where(tc <= tP1, (0.5*tc**2*acc1),
                    np.where(tc <= tP2, (tc-tP1)*vExt + 0.5*tP1**2*acc1,
                        np.where(tc <= tP4, 0.5*(tc-tP3)**2*acc2 - 0.5*(tP2-tP3)**2*acc2 + (tP2-tP1)*vExt + 0.5*tP1**2*acc1, 
                                 np.where(tc <= tP5, (tc-tP4)*vFlx + 0.5*(tP4-tP3)**2*acc2 - 0.5*(tP2-tP3)**2*acc2 + (tP2-tP1)*vExt + 0.5*(tP1)**2*acc1,
                                          np.where(tc<=tP6, 0.5*(tc-tP6)**2*acc1 - 0.5*(tP5-tP6)**2*acc1 + (tP5-tP4)*vFlx + 0.5*(tP4-tP3)**2*acc2 -0.5*(tP2-tP3)**2*acc2 + (tP2-tP1)*vExt + 0.5*(tP1)**2*acc1, 0)))))
    phi = phi+phiAvg+amp
    phid = None
    phidd = None
    
    return phi, phid, phidd

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
