# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:18:59 2025

@author: David
"""

import concurrent.futures
import numpy as np
from scipy import optimize

#from FuncMus import ForceEQ, SolveSimuMTC
from hillmodel import ForceEQ, SolveSimuMTC
from AnalyticalGamma import AnlyGamma

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gridRef(stim,cf,fts,kje,phiAvg,acc,muspar,funcMotion):
    
    if stim == 2:
        bnds = ((-np.inf, np.inf), (0,np.inf),)
        x0 = [0, fts/cf*0.2*np.random.normal(1,1e-2*fts/cf)]
    elif stim == 1:
        bnds = ((0,np.inf),)
        x0 = [fts/cf*0.2*np.random.normal(1,1e-2*fts/cf)]
    
    fun = lambda x: -simPeriodic(x,cf,fts,kje,phiAvg,acc,muspar,funcMotion)[0]
    result = optimize.minimize(fun,x0,method='Nelder-Mead',bounds=bnds,options={'xatol': 1e-3, 'fatol': 1e-2})
    x = result.x
    
    Pmech, y = simPeriodic(x,cf,fts,kje,phiAvg,acc,muspar,funcMotion)

    return Pmech, y

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def simPeriodic(tStim,cf,fts,kje,phiAvg,acc,muspar,funcMotion):   
    if len(tStim)>1:
        tStimOn = tStim[0]
        tStimOff = tStim[1]
    else:
        tStimOn = 0
        tStimOff = tStim[0]
        
    N = 2000
    time = np.unique(np.hstack((np.arange(0,fts/cf,1/N), np.arange(fts/cf,1/cf,1/N),1/cf)))
    # breakpo1int()
    phi     = funcMotion(time, cf, fts, kje, phiAvg, acc)[0]
    lmtc    = phi*muspar['A1']+muspar['A0'] 
    
    # gamma0 = muspar['gamma_0']
    gamma0 = AnlyGamma(0, cf, tStimOn, tStimOff, 1, muspar)[0]
    lcerel0 = np.min([1.4, ForceEQ(lmtc[0],gamma0,muspar)[1]-1e-2])
    lcerel0_Initial = lcerel0 # for debugging
    
    # Create dict as input for functions
    solmat = {}
    solmat['time'] = time
    solmat['phi'] = phi
    solmat['tStim'] = [tStimOn, tStimOff]
    solmat['cf'] = cf

    # Simulate until difference in SEE force is <10 mN
    dFsee, dLcerel = 1000, 1
    iRound, iFail = 0, 0
    timeout = 10 # [s]

    def solve_ode(gamma0,lcerel0,ode_opts,solmat):
        Wmech, y = SolveSimuMTC(gamma0,lcerel0,muspar,solmat,ode_opts)
        
        time, _, _, _, lcerel, _, _, _, _, fsee, *_ = y
        
        # Check if solution is complete and within bounds
        if time[-1] != ode_opts['t_eval'][-1] or lcerel[-1] > 2:
            raise RuntimeError("Incomplete simulation or lcerel blew up")
        
        return Wmech, y
    
    ode_opts = {}
    if lcerel0 < 1.4:
        ode_opts['method'] = 'LSODA'
        ode_opts['rtol'] = 1e-6
        ode_opts['atol'] = 1e-3
    else: # we might run into problem if the solver is not stiff enough for these sims!
        ode_opts['method'] = 'Radau'
        ode_opts['rtol'] = 1e-9
        ode_opts['atol'] = 1e-6
    
    ode_opts['t_eval'] = time
    lcerel_f = [lcerel0]
       
    # Sometimes a simulation does get stuck, so if it takes longer than 10s we abort it and try again with a almost identical initial state.
    while dFsee > muspar['fmax']*0.1/100 or abs(dLcerel) > 1e-3:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Using ThreadPoolExecutor to run solve_ivp with timeout
            future = executor.submit(solve_ode,gamma0,lcerel0,ode_opts,solmat) 
            try:
                # Wait for the result with a timeout
                Wmech,y = future.result(timeout=timeout)
                time, phi, stim, gamma, lcerel, q, lmtc, lsee, lpee, fisomrel, fsee, fpee, fce, fcerel, vcerel = y
                dFsee = np.abs(fsee[0]-fsee[-1])
                dLcerel = lcerel[-1]-lcerel[0]
                lcerel_f.append(lcerel[-1])
                
                # sometimes we have large outliers, select the one within 2 std.
                lcerel_sel = np.array(lcerel_f[-4:]) # select last 4 values
                lcerel_sel = lcerel_sel[np.abs(lcerel_sel - np.mean(lcerel_sel)) <= 2*np.std(lcerel_sel)]
                lcerel0 = np.mean(lcerel_sel)  # avg.
            except:
                print(f"Timeout at iRound {iRound}, trying again")
                # breakpoint()
                lcerel0 -= 0.04
                iFail += 1
                ode_opts['method'] = 'Radau'
                ode_opts['rtol'] = 1e-9
                ode_opts['atol'] = 1e-6
            finally:
                # Use shutdown(wait=False) to avoid blocking while cleaning up
                executor.shutdown(wait=False)
  
            iRound += 1
            if iRound == 20: # if we're here than we might need to solve more accurately
                ode_opts['method'] = 'Radau'
                ode_opts['rtol'] = 1e-9
                ode_opts['atol'] = 1e-6
            if iRound > 40 or iFail > 3:
                if Wmech<0: # it's a shit stim(t)!
                    dFsee, dLcerel, Wmech, y = 0,0,0,0
                    continue
                else:
                    breakpoint()
                    
    
    Pmech = Wmech*cf
    print(f'AMPO = {Pmech:0.1f} W')
    # print(f'tStimOff = {tStimOff:0.6f} s')
    return Pmech, y