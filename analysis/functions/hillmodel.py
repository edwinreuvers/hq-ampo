"""
This module contains the functions of a Hill-type muscle model,
including activation dynamics, force–length and force–velocity relations.

This model is extensively described in the following papers:
    -   van Soest, A.J.K. & Bobbert (1993) 
        The contribution of muscle properties in the control of explosive 
        movements
        https://doi.org/10.1007/BF00198959
    -   Reuvers, E.D.H.M. & Kistemaker, D.A. (2025)
        Accuracy of experimentally estimated muscle properties: Evaluation and 
        improvement using a newly developed toolbox
        https://doi.org/10.1101/2025.09.29.678508 
    -   
"""

import numpy as np
from scipy.integrate import solve_ivp, trapezoid

def simu_mtc(t, state, muspar, inputs):
    """
    Compute state derivatives for a Hill-type muscle-tendon complex (MTC) model.

    Parameters
    ----------
    t : float
        Current time [s].
    state : array-like, shape (2,)
        Current state vector:
        - state[0] : gamma, normalized Ca2+ concentration between filaments.
        - state[1] : lcerel, relative contractile element (CE) length
                     (lce / lce_opt).
    muspar : dict
        Muscle parameters including gamma dynamics, CE properties, etc.
    inputs : dict
        Input containing:
        - 'time': time axis [s]
        - 'phi': joint/muscle angle trajectory
        - 'stim' or 't_stim': stimulation over time or onset/offset times

    Returns
    -------
    gammad : float
        Time derivative of gamma [1/s].
    vcerel : float
        Time derivative of relative CE length [1/s].
    y : list
        List of additional variables for debugging/analysis:
        [phi, stim, q, l_mtc, lsee, lpee, fisomrel, fsee, fpee, fce,
        fcerel, vcerel]
    """
    
    gamma, lcerel = state

    # Muscle-tendon length
    phi = np.interp(t, inputs['time'], inputs['phi'])
    lmtc = phi * muspar['A1'] + muspar['A0']

    # Determine stimulation
    try:
        t_stim = np.atleast_2d(inputs['t_stim'])
        stim = np.zeros_like(t, dtype=float)
        for start, end in t_stim:
            stim[(t >= start) & (t <= end)] = 1.0
    except KeyError:
        stim = np.interp(t, inputs['time'], inputs['stim'])

    # Activation dynamics
    gamma_0 = muspar['gamma_0']
    gamma = (gamma>gamma_0)*gamma + (gamma<=gamma_0)*gamma_0 # [ ]
    q = act_state(gamma, lcerel, muspar)[0]
    gammad = np.where(
        stim >= gamma,
        (stim * (1 - gamma_0) - gamma + gamma_0) / muspar['tact'],
        (stim * (1 - gamma_0) - gamma + gamma_0) / muspar['tdeact']
    )

    # Contraction dynamics
    lce = lcerel * muspar['lce_opt']
    lsee = lmtc - lce
    lpee = lce
    fisomrel = force_length(lcerel, muspar)[0]
    fsee, fpee = lee2force(lsee, lpee, muspar)[0:2]
    fce = fsee - fpee
    fcerel = fce / muspar['fmax']
    vcerel = fce2vce(fce, q, lcerel, muspar)[1]

    # Debugging
    if np.isnan(vcerel).any():
        breakpoint()

    y = [phi, stim, q, lmtc, lsee, lpee, fisomrel, fsee,
         fpee, fce, fcerel, vcerel]

    return gammad, vcerel, y

def solve_simu_mtc(gamma0, lcerel0, muspar, inputs, ode_opts=None):
    """
    Forward simulation of a Hill-type muscle-tendon complex (MTC) model.

    Parameters
    ----------
    gamma0 : float
        Initial normalized Ca2+ concentration between filaments [ ].
    lcerel0 : float
        Initial relative CE length (lce / lce_opt) [ ].
    muspar : dict
        Muscle parameters dictionary.
    inputs : dict
        Inputs containing time axis, lmtc(t), stim(t) or t_stim.
    ode_opts : dict, optional
        ODE solver options:
        - 'method' : str, solver method (default: 'Radau')
        - 'max_step' : float, maximum step size
        - 'rtol' : float, relative tolerance
        - 'atol' : float, absolute tolerance
        - 't_eval' : array-like, time points to evaluate solution

    Returns
    -------
    W_mech : float
        Mechanical work done by the SEE [J].
    y : list
        List of simulation outputs: [time, phi, stim, gamma, lce_rel, q,
        l_mtc, l_see, l_pee, f_isom_rel, f_see, f_pee, f_ce, f_ce_rel, v_ce_rel]
    """
    if ode_opts is None:
        ode_opts = {}
    
    # Solver options
    method = ode_opts.get('method', 'Radau')
    max_step = ode_opts.get('max_step', np.inf)
    rtol = ode_opts.get('rtol', 1e-3)
    atol = ode_opts.get('atol', 1e-6)
    t_eval = ode_opts.get('t_eval', inputs['time'])

    # Initial state and timespan
    state0 = [gamma0, lcerel0]
    t_span = [inputs['time'][0], inputs['time'][-1]]

    # ODE function wrapper
    def ode_fun(t, state):
        gammad, vcerel, _ = simu_mtc(t, state, muspar, inputs)
        return [gammad, vcerel]

    # Solve ODE
    sol = solve_ivp(
        ode_fun, t_span, state0,
        method=method, max_step=max_step,
        rtol=rtol, atol=atol, t_eval=t_eval
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    # Evaluate solution at all time points
    gammad, vcerel, y_list = simu_mtc(sol.t, sol.y, muspar, inputs)

    phi = y_list[0]
    stim = y_list[1]
    q = y_list[2]
    lmtc = y_list[3]
    lsee = y_list[4]
    lpee = y_list[5]
    fisomrel = y_list[6]
    fsee = y_list[7]
    fpee = y_list[8]
    fce = y_list[9]
    fcerel = y_list[10]
    vcerel = y_list[11]

    y = [sol.t, phi, stim, sol.y[0], sol.y[1], q, lmtc, lsee, lpee,
         fisomrel, fsee, fpee, fce, fcerel, vcerel]

    # Mechanical work by SEE
    W_mech = -trapezoid(fsee, lmtc)
    return W_mech, y

def gamma_dot(gamma, stim, muspar):
    """
    Compute rate of change of Ca²⁺ concentration.
    
    Parameters
    ----------
    gamma : float or numpy.ndarray
        Current Ca²⁺ level [-].
    stim : float or numpy.ndarray
        Neural stimulation [-].
    muspar : dict
        Parameters: gamma_0, tact, tdeact
    
    Returns
    -------
    gamma_dot : float or numpy.ndarray
        Time derivative of gamma [1/s].
    """
    
    # Unravel parameter values
    gamma_0 = muspar['gamma_0']
    tact    = muspar['tact']
    tdeact  = muspar['tdeact']
    
    # Computations
    gammadot = (stim>=gamma)*((stim*(1-gamma_0)-gamma + gamma_0)/tact) + (stim<gamma)*((stim*(1-gamma_0)-gamma + gamma_0)/tdeact) # [1/s]
    
    # Output
    return gammadot

def act_state(gamma, lcerel, muspar):
    """
    Computes the active state based on the relative amount of Ca2+ 
    between the filaments and the relative CE length. This version is based on 
    Hatze (1981), p. 37-41, but slightly modified such that the parameter
    values has physiological relevance and that it is easier to use in 
    Optimal Control.
        
    Parameters
    ----------
    gamma : float or numpy.ndarray
        Relative amount Ca2+ between the filaments [-]
    lcerel : float or numpy.ndarray
        Relative CE length (Lce / Lce_opt) [-].
    muspar : dict
        Muscle parameters:
            q0, kCa, a_act, b_act
    
    Returns
    -------
    q : float or numpy.ndarray
        Active state (relative amount of Ca2+ bound to troponin C) [-].
    dqdlcerel : float or numpy.ndarray
        Partial derivative of q with respect to lcerel [-].
    gamma_05 : float or numpy.ndarray
        Gamma value at which q = 0.5 [-].
    """
    
    # Unravel parameters values
    q0 = muspar['q0']
    kCa = muspar['kCa']
    a_act = muspar['a_act']
    b_act = muspar['b_act']
    
    # Computations
    a1_act = np.log10(np.exp(a_act))
    B_act = b_act[0] + b_act[1] * lcerel + b_act[2] * lcerel**2
    
    q = q0 + (1 - q0) / (1 + (kCa * gamma)**a1_act * np.exp(a_act * B_act))
    
    try:
        dqdlcerel = (a_act*np.exp(a_act*(B_act))*(gamma*kCa)**(np.log(np.exp(a_act))/np.log(10))*(q0-1)*(b_act[1] + 2*b_act[2]*lcerel))/(np.exp(a_act*(B_act))*(gamma*kCa)**(np.log(np.exp(a_act))/np.log(10)) + 1)**2
        gamma_05 = ((1-0.5)/(kCa**a1_act*np.exp(a_act*B_act)*(0.5-q0)))**(1/a1_act)
    except:
        dqdlcerel = np.nan*gamma
        gamma_05 = np.nan*gamma
    
    return q, dqdlcerel, gamma_05

def force_length(lcerel, muspar):
    """
    Computes the relative isometric CE force based on the relative CE length.
    
    Parameters
    ----------
    lcerel : float or numpy.ndarray
        Relative CE length (Lce / Lceopt) [-].
    muspar : dict
        Muscle parameters.
    
    Returns
    -------
    fisomrel : float or numpy.ndarray
        Relative isometric CE force (CE isometric force / Fcemax) [-].
    kce : float or numpy.ndarray
        Derivative fisomrel with respect to lcerel [-].
    """
    
    # Unravel parameter values
    n = muspar['n'] # [ ]
    C = -1/muspar['w']**n # [ ]
    
    # Compute tails of parabola (exponential tails)
    Fp = 0.1 # exp function kicks in a 10% fisomrel
    xp = ((Fp-1)/C)**(1/2) # intercept of function with y=Fp
    xp = np.array([-xp+1, xp+1]) # two solutions because of root
    dFdx = 2*C*(xp-1) # first derivative of F at xp
    # function has the form: y=a*exp(b*x), so:
    b = dFdx/Fp
    a = Fp/(np.exp(b*xp))
    
    # Compute fisomrel
    fisomrel = np.clip(
        np.piecewise(
            lcerel, 
            [lcerel < xp[0], (lcerel >= xp[0]) & (lcerel <= xp[1]), lcerel > xp[1]], 
            [lambda x: a[0]*np.exp(b[0]*x),           # Exponential tail for lcerel < xp[0]
             lambda x: C*(x-1)**n + 1,                # Middle section
             lambda x: a[1]*np.exp(b[1]*x)]           # Exponential tail for lcerel > xp[1]
        ),
        1e-9, # Minimum value
        None  # No maximum value
    )
    
    # Compute derivative (i.e., fisomrel/dlcerel)
    kce = np.piecewise(lcerel, 
                        [lcerel < xp[0], (lcerel >= xp[0]) & (lcerel <= xp[1]), lcerel > xp[1]], 
                        [lambda x: a[0]*b[0]*np.exp(b[0]*x), # [ ] dfisomrel/dlcerel for lcerel < xp[0]
                         lambda x: 2*C*(x-1), # [ ] dfisomrel/dlcerel
                         lambda x: a[1]*b[1]*np.exp(b[1]*x)]) # [ ] dfisomrel/dlcerel for lcerel < xp[0]
    
    return fisomrel,kce

def lee2force(lsee, lcerel, muspar):
    """
    Compute forces in SEE and PEE.
    
    Parameters
    ----------
    lsee : float or numpy.ndarray
        SEE length [m]
    lcerel : float or numpy.ndarray
        Relative CE length (Lce / Lceopt) [-]
    muspar : dict
        Muscle parameters
    
    Returns
    -------
    fsee : float or numpy.ndarray
        SEE force [N]
    fpee : float or numpy.ndarray
        PEE force [N]
    fce : float or numpy.ndarray
        CE force [N]
    fcerel : float or numpy.ndarray
        CE force normalised to maximal isometric CE force [-]
    Ksee : float or numpy.ndarray
        dFsee/dLsee [N/m]
    Kpee : float or numpy.ndarray
        dFPee/dLpee [N/m]
    lsee : float or numpy.ndarray
        SEE length [m]
    lpee : float or numpy.ndarray
        PEE length [m]
    """
    
    # Unravel parameter values
    lce_opt = muspar['lce_opt'] # [m]       CE optimum length (i.e., CE length at which isometric CE force is maximal)
    lpee0   = muspar['lpee0']   # [m]       PEE slack length
    kpee    = muspar['kpee']    # [N/m^2]   shape parameter that scales the stiffness of PEE
    lsee0   = muspar['lsee0']   # [m]       SEE slack length
    ksee    = muspar['ksee']    # [N/m^2]   shape parameter that scales the stiffness of SEE
    fmax    = muspar['fmax']    # [N]       maximal isometric CE force
    
    # SEE
    esee    = lsee-lsee0 # [m] SEE elongation
    fsee    = (esee<0)*0 + (esee>=0)*(ksee*esee**2) # [N] SEE force
    Ksee    = (esee<0)*0 + (esee>=0)*(2*ksee*esee) # [N/m] dFsee/dLsee
    
    # PEE
    lpee    = lcerel*lce_opt # [m] PEE length
    epee    = lpee-lpee0 # PEE elongation
    fpee    = (epee<0)*0 + (epee>=0)*(kpee*epee**2) # [N] PEE force
    Kpee    = (epee<0)*0 + (epee>=0)*(2*kpee*epee) # [N/m] dFpee/dLpee
    
    # CE
    fce     = fsee-fpee # [N]
    fcerel  = fce/fmax # [ ]
    
    # Output
    return fsee,fpee,fce,fcerel,Ksee,Kpee,lsee,lpee

def fce2vce(fce, q, lcerel, muspar):
    """
    Compute CE velocity from force–velocity relationship.
    
    Parameters
    ----------
    fce : float or numpy.ndarray
        Contractile element force [N]
    q : float or numpy.ndarray
        Active state [-]
    lcerel : float or numpy-array
        Relative CE length [-]
    muspar : dict
        Muscle parameters
    
    Returns
    -------
    vce : float or numpy.ndarray
        CE velocity [m/s]
    vcerel : float or numpy.ndarray
        Relative CE velocity (CE velocity / Lceopt) [1/s]
        Note: computed only if optimum CE length is known, else value is None
    """
    
    # Unravel muscle parameters
    a_c, b_c            = muspar['a'], muspar['b']
    fasymp, fmax        = muspar['fasymp'], muspar['fmax']
    slopfac, vfactmin   = muspar['slopfac'], muspar['vfactmin']
    q0                  = muspar['q0']  
    sloplin = vfactmin*b_c/(slopfac*0.005*0.0975*(fmax+a_c))
    
    # Scale arel and brel if necessary
    fisomrel = force_length(lcerel,muspar)[0]
    
    # Smooth version of KvS brel(q)
    q0_b = (np.log(1/vfactmin-1)+q0*22)/22
    b = b_c/(1+np.exp(-22*(q-q0_b)))
    
    # Scale a
    a = a_c
    a = (lcerel>1)*a*fisomrel + (lcerel<=1)*a

    # Variables for various part of vce-fce relation
    dvdf_isom_con = b/(q*(fisomrel*fmax+a)) # slope in the isometric point at wrt concentric part
    dvdf_isom_ecc = dvdf_isom_con/slopfac # slope in the isometric point at wrt eccentric part
    dFdvcon0      = 1/dvdf_isom_con
    s_as          = 1/sloplin
    p1 = -(fisomrel*q*(fasymp*fmax - fmax))/(s_as - dFdvcon0*slopfac) 
    p2 =  (fisomrel**2*q**2*(fasymp*fmax - fmax)**2)/(s_as - dFdvcon0*slopfac)
    p3 =  -fasymp*fisomrel*q*fmax;
    p4 =  -s_as

    # Compute different regions
    r_c1 = (((fce/q) <= fisomrel*fmax) * (dvdf_isom_con<=sloplin)) # Concentric, dvdf_isom_con<=sloplin (normal)	
    r_c2 = (((fce/q) <= fisomrel*fmax) * (dvdf_isom_con>sloplin)) # Concentric dvdf_isom_con>sloplin (defective case)
    r_e1 = (((fce/q) > fisomrel*fmax) * (dvdf_isom_ecc<=(sloplin/slopfac))) # Eccentric, dvdf_isom_ecc<=sloplin (normal) 
    r_e2 = (((fce/q) > fisomrel*fmax) * (dvdf_isom_ecc>(sloplin/slopfac))) # Eccentric, dvdf_isom_ecc>sloplin (defective case)
    
    #  Compute CE velocity
    vce_c1 = (b*(fce-q*fisomrel*fmax)/(fce+q*a)) # Concentric, dvdf_isom_con<=sloplin (normal)	
    vce_c2 = (sloplin*(fce-q*fisomrel*fmax)) # Concentric dvdf_isom_con>sloplin (defective case)        
    vce_e1 = ((-(fce + p3 + p1*p4 + (fce**2 - 2*fce*p1*p4 + 2*fce*p3 + p1**2*p4**2 - 2*p1*p3*p4 + p3**2 + 4*p2*p4)**(1/2))/(2*p4))) # Eccentric, dvdf_isom_ecc<=sloplin (normal)
    vce_e2 = ((sloplin/slopfac)*(fce-q*fisomrel*fmax)) # Eccentric, dvdf_isom_ecc>sloplin (defective case)
    
    # Output   
    vce = r_c1*vce_c1 + r_c2*vce_c2 + r_e1*vce_e1 + r_e2*vce_e2        
        
    if 'lce_opt' in muspar:
        vcerel = vce/muspar['lce_opt']
    else:
        vcerel = None
    
    return vce, vcerel, [vce_c1, vce_c2, vce_e1, vce_e2], [r_c1, r_c2, r_e1, r_e2]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def force_eq(lmtc,gamma,muspar):
    """
    force_eq Finds relative CE length such that SEE force equals the sum of CE
        and PEE force, for any given MTC-length and gamma (i.e., normalised
        concentration Ca2+ between the filaments)
    
    Parameters
    ----------
    lmtc : float or numpy.ndarray
        muscle-tendon-complex length [m]
    gamma : float or numpy.ndarray
        normalised concentration Ca2+ between the filaments [-]
    muspar : dict
        Muscle parameters
    
    Returns
    -------
    fsee : float or numpy.ndarray
        SEE force [N]
    lcerel : float or numpy.ndarray
        relative CE length, i.e. CE length divided by optimum CE length) [-]
    fce : float or numpy.ndarray
        CE force [N]
    fpee : float or numpy.ndarray
        PEE force [N]
    """
        
    # Unravel parameter values
    lce_opt = muspar['lce_opt']     # [m] CE optimum length (i.e., CE length at which isometric CE force is maximal)
    lsee0   = muspar['lsee0']       # [m] SEE slack length
    fmax    = muspar['fmax']        # [N] maximal CE force
    C       = -1/muspar['w']**2     # [ ] parameter based on width of 2nd order polynomial of isometric CE force-length relation
    ksee    = muspar['ksee']        # [N/m^2] shape parameter that scales the stiffness of SEE
    
    # Get initial guess of relative CE length   
    q = (gamma>muspar['q0'])*act_state(gamma,1,muspar)[0] + (gamma<=muspar['q0'])*muspar['q0'] # [ ]
    # lcerel =  -(ksee*lce_opt*lsee0 - ksee*lce_opt*lmtc + fmax**(1/2)*q**(1/2)*(ksee*C*lce_opt**2 + ksee*C*lmtc**2 + ksee*C*lsee0**2 + ksee*lce_opt**2 - C*fmax*q - 2*ksee*C*lce_opt*lmtc + 2*ksee*C*lce_opt*lsee0 - 2*ksee*C*lmtc*lsee0)**(1/2) + C*fmax*q)/(ksee*lce_opt**2 - C*fmax*q)  # [ ]
    a = ksee*lce_opt*lsee0 - ksee*lce_opt*lmtc + C*fmax*q
    b1 = fmax**(1/2)*q**(1/2)
    b2 = ksee*C*lce_opt**2 + ksee*C*lmtc**2 + ksee*C*lsee0**2 + ksee*lce_opt**2 - C*fmax*q - 2*ksee*C*lce_opt*lmtc + 2*ksee*C*lce_opt*lsee0 - 2*ksee*C*lmtc*lsee0
    b2 = (b2>=0)*b2 + (b2<0)*0   
    b = b1*b2**(1/2) 
    c = ksee*lce_opt**2 - C*fmax*q
    lcerel = -(a+b)/c
    
    # Set values for first round
    fce, fpee, fsee, dlcerel = 1e6, 0, 0, 0
    # Set tolerance
    tolF = 1e-5*muspar['fmax'] # until 0.01% of Fmax
    
    # Newton root finding to find relative CE length
    while (np.max(np.abs(fce+fpee-fsee))>tolF):
        lcerel = lcerel+dlcerel  # [ ]
        fisomrel,Kcerel = force_length(lcerel,muspar)  # [ , ]
        q,Kq = act_state(gamma,lcerel,muspar)[0:2] # [ , ]
        fce = q*fisomrel*muspar['fmax'] # [N]
        Kce = fisomrel*Kq*muspar['fmax']+q*Kcerel*muspar['fmax'] # [N]
        lce = lcerel*muspar['lce_opt']
        lsee = lmtc-lce
        fsee,fpee,_,_,Ksee,Kpee = lee2force(lsee,lcerel,muspar)[0:6] # [N, N, N/m, N/m]
        Ksee = -Ksee*muspar['lce_opt'] # [N]
        Kpee = Kpee*muspar['lce_opt'] # [N]
        dlcerel = (fce+fpee-fsee)/(Ksee-Kce-Kpee) # [ ]
        pass
    
    # Output
    return fsee,lcerel,fce,fpee

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def anly_gamma(time, cf, t_stim_on, t_stim_off, stim, muspar):
    """
    Calculate gamma over time for constant muscle stimulation with periodic behavior.

    Parameters
    ----------
    time : np.ndarray
        Time axis.
    cf : float
        Cycle frequency in Hz.
    t_stim_on : float
        Time at which muscle stimulation switches to the constant value of `stim`.
    t_stim_off : float
        Time at which muscle stimulation switches to 0.
    stim : float
        Value of constant muscle stimulation.
    muspar : dict
        Dictionary containing:
        - 'gamma_0': Minimum value of gamma.
        - 'tact': Time constant for activation.
        - 'tdeact': Time constant for deactivation.

    Returns
    -------
    gamma : np.ndarray
        Gamma as a function of time.
    stim_t : np.ndarray
        Stimulation as a function of time.
    """
    gamma_0 = muspar['gamma_0']
    tact = muspar['tact']
    tdeact = muspar['tdeact']

    # Cycle duration
    Tcycle = 1 / cf

    # Shift time with t_stim_on and take modulus
    time_mod = np.mod(time - t_stim_on, Tcycle)
    t_stim_off_mod = np.mod(t_stim_off - t_stim_on, Tcycle)
    t_stim_on_mod = 0

    # Calculate gamma at t=0 for periodic behavior
    gamma0_1 = (
        gamma_0 +
        np.exp(-Tcycle / tdeact) *
        np.exp(t_stim_off_mod / tdeact) *
        (gamma_0 - 1) *
        (stim * np.exp(-t_stim_off_mod / tact) - stim +
         (gamma_0 * stim * np.exp(-t_stim_off_mod / tact)) / (stim - gamma_0 * stim))
    ) / (
        (stim * np.exp(-Tcycle / tdeact) *
         np.exp(-t_stim_off_mod / tact) *
         np.exp(t_stim_off_mod / tdeact) *
         (gamma_0 - 1)) / (stim - gamma_0 * stim) + 1
    )

    # Calculate gamma at the end of stimulation phase
    gamma0_2 = gamma_rise(t_stim_off_mod, gamma0_1, stim, muspar)

    # Compute gamma(t)
    gamma = np.where(
        (time_mod >= 0) & (time_mod < t_stim_off_mod),
        gamma_rise(time_mod, gamma0_1, stim, muspar),
        gamma_relax(time_mod - t_stim_off_mod, gamma0_2, muspar)
    )

    # Compute stimulation over time
    stim_t = np.where((time_mod >= 0) & (time_mod < t_stim_off_mod), stim, 0)

    return gamma, stim_t


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gamma_relax(time, gamma0, muspar):
    """
    Calculate gamma over time during relaxation (stim = 0).

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    gamma0 : float
        Initial gamma at t=0.
    muspar : dict
        Dictionary containing:
        - 'gamma_0': Minimum value of gamma.
        - 'tdeact': Time constant for deactivation.

    Returns
    -------
    gamma : np.ndarray
        Relative amount of Ca2+ between filaments during relaxation.
    """
    gamma_0 = muspar['gamma_0']
    tdeact = muspar['tdeact']

    t_shift = -np.log((gamma_0 - gamma0) / (gamma_0 - 1)) * tdeact
    gamma = (1 - gamma_0) * np.exp(-(time + t_shift) / tdeact) + gamma_0

    return gamma


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gamma_rise(time, gamma0, stim, muspar):
    """
    Calculate gamma over time during activation (stim > 0).

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    gamma0 : float
        Initial gamma at t=0.
    stim : float
        Normalized muscle stimulation.
    muspar : dict
        Dictionary containing:
        - 'gamma_0': Minimum value of gamma.
        - 'tact': Time constant for activation.

    Returns
    -------
    gamma : np.ndarray
        Relative amount of Ca2+ between filaments during activation.
    """
    gamma_0 = muspar['gamma_0']
    tact = muspar['tact']

    t_shift = -np.log((gamma_0 - gamma0) / (stim - gamma_0 * stim) + 1) * tact
    gamma = stim * (1 - gamma_0) * (1 - np.exp(-(time + t_shift) / tact)) + gamma_0

    return gamma
