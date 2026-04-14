"""
Module for simulating periodic muscle-tendon complex (MTC) dynamics
under different muscle stimulation protocols using Hill-type models.

Functions
---------
grid_ref(stim, cf, fts, kje, phi_avg, acc, muspar, func_motion)
    Optimizes the stimulation timing to maximize mechanical power.
    
sim_periodic(t_stim, cf, fts, kje, phi_avg, acc, muspar, func_motion)
    Simulates a single periodic cycle of MTC dynamics.
"""

import concurrent.futures
import numpy as np
from scipy import optimize

import hillmodel

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def optimise_stim(stim, cf, fts, kje, phi_avg, acc, muspar, func_motion):
    """
    Optimise stimulation onset and offset to maximize mechanical power.

    Parameters
    ----------
    stim : int
        Type of stimulation (1 = single pulse, 2 = constant). Determines bounds.
    cf : float
        Cycle frequency [Hz].
    fts : float
        Fraction of cycle used for stimulation.
    kje : float
        Joint-specific parameter for motion function.
    phi_avg : float
        Average joint angle or position.
    acc : float
        Acceleration parameter for motion function.
    muspar : dict
        Muscle parameters dictionary including gamma parameters.
    func_motion : callable
        Function returning motion profile (phi, lmtc) given time, cf, fts, etc.

    Returns
    -------
    p_mech : float
        Mechanical power output [W].
    sim_result : tuple
        Full simulation results from sim_periodic.
    """
    # Define bounds and initial guess
    if stim == 2:
        bounds = ((-np.inf, np.inf), (0, np.inf))
        x0 = [0, fts / cf * 0.2]
    elif stim == 1:
        bounds = ((0, fts / cf),)
        x0 = [fts / cf * 0.2]
            
    # Objective: negative mechanical power
    objective = lambda x: -sim_periodic(x, cf, fts, kje, phi_avg, acc, muspar, func_motion)[0]

    result = optimize.minimize(
        objective,
        x0,
        method='Nelder-Mead',
        bounds=bounds,
        options={'xatol': 1e-5, 'fatol': 1e-3}
    )

    t_stim_opt = result.x
    p_mech, sim_result = sim_periodic(t_stim_opt, cf, fts, kje, phi_avg, acc, muspar, func_motion)

    return p_mech, sim_result

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def sim_periodic(t_stim, cf, fts, kje, phi_avg, acc, muspar, func_motion):
    """
    Simulate a single periodic cycle of muscle-tendon complex (MTC) dynamics.

    Parameters
    ----------
    t_stim : array-like or float
        Stimulation onset and offset times. If single value, onset = 0.
    cf : float
        Cycle frequency [Hz].
    fts : float
        Fraction of cycle used for stimulation.
    kje : float
        Joint-specific parameter for motion function.
    phi_avg : float
        Average joint angle or position.
    acc : float
        Acceleration parameter for motion function.
    muspar : dict
        Muscle parameters dictionary including gamma parameters.
    func_motion : callable
        Function returning motion profile (phi, lmtc) given time, cf, fts, etc.

    Returns
    -------
    p_mech : float
        Mechanical power output [W].
    sim_result : tuple
        Detailed simulation results.
    """
    
    # Determine stimulation times
    if len(t_stim) > 1:
        t_stim_on, t_stim_off = t_stim
    else:
        t_stim_on, t_stim_off = 0, t_stim[0]
    
    # Time discretization
    n_points = 2000
    time = np.unique(np.hstack((
        np.arange(0, fts / cf, 1 / n_points),
        np.arange(fts / cf, 1 / cf, 1 / n_points),
        [1 / cf]
    )))
    
    # Knee jont motion over time and correspond MTC length
    phi = func_motion(time, cf, fts, kje, phi_avg, acc)[0]
    lmtc = phi * muspar['A1'] + muspar['A0']
    
    # Initial states (gamma and lcerel) at t=0
    gamma0 = hillmodel.anly_gamma(0, cf, t_stim_on, t_stim_off, 1, muspar)[0]
    lcerel0 = min(1.4, hillmodel.force_eq(lmtc[0], gamma0, muspar)[1] - 1e-2)
    lcerel_f = [lcerel0]
    
    # Setup solution dictionary
    inputs = {
        'time': time,
        'phi': phi,
        't_stim': [t_stim_on, t_stim_off],
        'cf': cf
    }

    # Convergence parameters
    dFsee, dLcerel = 1000, 1
    iRound, iFail = 0, 0
    timeout = 10  # seconds

    # Simulate until difference in SEE force is <10 mN
    dFsee, dLcerel = 1000, 1
    iRound, iFail = 0, 0
    timeout = 10 # [s]

    # ODE solver wrapper
    def solve_ode(gamma0, lcerel0, ode_opts, u):
        W_mech, sim_result = hillmodel.solve_simu_mtc(gamma0, lcerel0, muspar, u, ode_opts)
        
        # Extract key variables
        time, _, _, _, lcerel, _, _, _, _, fsee, *_ = sim_result
        
        # Check if solution is complete and within bounds
        if time[-1] != ode_opts['t_eval'][-1] or lcerel[-1] > 2:
            raise RuntimeError("Incomplete simulation or lcerel blew up")
        
        return W_mech, sim_result
    
    
    ode_opts = {}   
    ode_opts['method'] = 'Radau'
    ode_opts['rtol'] = 1e-9
    ode_opts['atol'] = 1e-6
    ode_opts['t_eval'] = time
    lcerel_f = []
    
    # DEBUGGING REMOVE
    hillmodel.solve_simu_mtc(gamma0, lcerel0, muspar, inputs, ode_opts)
    
    # Sometimes a simulation does get stuck, so if it takes longer than 10s we abort it and try again with a almost identical initial state.
    while dFsee > muspar['fmax']*0.1/100 or abs(dLcerel) > 1e-3:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Using ThreadPoolExecutor to run solve_ivp with timeout
            future = executor.submit(solve_ode,gamma0,lcerel0,ode_opts,inputs) 
            try:
                # Wait for the result with a timeout
                Wmech,y = future.result(timeout=timeout)
                time, phi, stim, gamma, lcerel, q, lmtc, lsee, lpee, fisomrel, fsee, fpee, fce, fcerel, vcerel = y
                dFsee = np.abs(fsee[0]-fsee[-1])
                dLcerel = lcerel[-1]-lcerel[0]
                lcerel_f.append(lcerel[-1])
                # lcerel0 = np.mean(lcerel_f[-3:])  # smooth over last 3 values
                
                # sometimes we have large outliers, select the one within 2 std.
                lcerel_sel = np.array(lcerel_f[-4:]) # select last 4 values
                lcerel_sel = lcerel_sel[np.abs(lcerel_sel - np.mean(lcerel_sel)) <= 2*np.std(lcerel_sel)]
                lcerel0 = np.mean(lcerel_sel)  # avg.
            except:
                print(f"Timeout at iRound {iRound}, trying again")
                lcerel0 -= 0.1
                iFail += 1
            finally:
                # Use shutdown(wait=False) to avoid blocking while cleaning up
                executor.shutdown(wait=False)
  
            iRound += 1
            if iRound > 10 or iFail > 3:
                dFsee, dLcerel, Wmech, y = 0,0,0,0
                breakpoint()
    
    Pmech = Wmech*cf
    print(f'AMPO = {Pmech:0.1f} W')
    return Pmech, y