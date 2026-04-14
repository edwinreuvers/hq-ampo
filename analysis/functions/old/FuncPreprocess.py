import numpy as np
import scipy.io as sio
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
import os
from FuncEMG import FiltEMG, MovAve, EnvEMG_Butter, EnvEMG_MovAve

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def preprocessData(dataDir,pp,day,cond,trial,fig=0):
    """
    Synchronization of knee dynonometer and stimula program, as well as
    computation for the 'compensated' torque (i.e., measured torque during
    active cycle minus those during passive cycles). 
    
    Inputs:
        pp    =   number of the participants
        day   =   number of measurement day (1, 2, 3 or 4)
        cond  =   number of the condition
        trial =   number of the trial (1 or 2)

    Outputs:
        data  =     a struct containing the time, knee joint angle, measured
                    torque, comparensated knee joint torque, emg of quad, emg 
                    of hams and the phase
    """
    
    #%% Load data
    # Define filename
    filename = f'pp{pp:02d}_day{day}_cond{cond:02d}_t{trial}'
    filename = os.path.join(dataDir,filename)
    
    # Load kinematics and kinetics from dino computer (stored in .mat-file)
    data = sio.loadmat(f"{filename}.mat")
    Tcycle = abs(data['Cfg']['CyclusDuration'][0][0]) / 1000  # [s]
    nr_cyc = int(data['Cfg']['NrCycli'][0][0][0][0])  # [ ]
    fs_dino = int(data['Cfg']['SampleFrequency'][0][0][0][0])  # [Hz]
    
    torque_dino = data['Data'][:, 1]                    # [Nm]
    time_dino = np.arange(len(torque_dino)) / fs_dino   # [s]
    phi_dino = np.deg2rad(data['Data'][:, 0])           # [rad]
    if pp == 5:
        phi_dino += (1.8 - np.max(phi_dino))  # Correct for the angle
    
    # Load emg data from stimula computer (stored in .csv-file)
    if os.path.isfile(f"{filename}.csv"):
        stila_data = pd.read_csv(f"{filename}.csv", header=None).T.to_numpy()
        fs_stila = 4000
        phi_stila = stila_data[0]  # [rad]
        torque_stila = stila_data[1]  # [Nm]
        emg_stila = stila_data[[3, 5]]  # [mV]
        time_stila = np.arange(len(phi_stila)) / fs_stila  # [s]
        
        # we have one file were the EMG did not get recorded (i.e., pp==3, cond 0.35D)
        # set the EMG to nan.
        if sum(emg_stila.flatten()==0) == len(emg_stila.T)*2: 
            emg_stila = emg_stila*np.nan
    else:
        stila_data = None
    
    #%% Time-syncronization (based on measured angle)
    # First 'upsample' knee angle from dino computer
    phi_dino_up = signal.resample(phi_dino, int(len(phi_dino)*fs_stila/fs_dino))
    
    x = phi_dino_up - np.mean(phi_dino_up)
    y = phi_stila - np.mean(phi_stila)
    # Compute cross-covariance with a max lag of 50 samples
    c, lags = xcov_np(x, y, max_lag=50)
    iMax = np.argmax(c)
    
    if lags[iMax] !=0: # They should be syncronized in time, else check what's going on!
        breakpoint()
    
    #%% Finding start and stop indices of cycles
    i_sf = signal.find_peaks(-phi_dino, distance=0.9*fs_dino*Tcycle)[0]
    i_se = signal.find_peaks(phi_dino, distance=0.9*fs_dino*Tcycle)[0]
    
    if i_sf[0] > 0.9*fs_dino*Tcycle: # if we didn't get the first index of the start of flexion!
        i_sf = np.insert(i_sf, 0, i_sf[0]-int(np.round(np.mean(np.diff(i_sf)))))
    
    i_sf = i_sf[:nr_cyc+1]
    i_se = i_se[:nr_cyc]
    
    if np.diff(i_sf)[-1] > 1.01*np.mean(np.diff(i_sf)): # is the last index is too far away that cannot be true!
        i_sf[-1] = i_sf[-2] + int(np.mean(np.diff(i_sf)))
        
    if stila_data is not None:
        j_sf = signal.find_peaks(-phi_stila, distance=0.9*fs_stila*Tcycle)[0]
        j_se = signal.find_peaks(phi_stila, distance=0.9*fs_stila*Tcycle)[0]
        
        if j_sf[0] > 0.9*fs_stila*Tcycle:
            j_sf = np.insert(j_sf, 0, 0)
    
        j_sf = j_sf[:nr_cyc+1]
        j_se = j_se[:nr_cyc]
        
        if np.diff(j_sf)[-1] > 1.01*np.mean(np.diff(j_sf)): # is the last index is too far away that cannot be true!
            j_sf[-1] = j_sf[-2] + int(np.mean(np.diff(j_sf)))
        
    # Plot data
    if fig == 1:
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.plot(time_dino, phi_dino, 'k', label="Knee Angle")
        plt.plot(i_sf / fs_dino, phi_dino[i_sf], 'rx', label="Start Cycles")
        plt.plot(i_se / fs_dino, phi_dino[i_se], 'rx', label="End Cycles")
        plt.xlabel("Time [s]")
        plt.ylabel("Knee Angle [rad]")
        plt.title("Knee Angle with Start and Stop Cycles")
        plt.legend()
    
        if stila_data is not None:
            plt.subplot(122)
            plt.plot(time_stila, phi_stila, 'k', label="Knee Angle (Stila)")
            plt.plot(j_sf / fs_stila, phi_stila[j_sf], 'rx', label="Start Cycles")
            plt.plot(j_se / fs_stila, phi_stila[j_se], 'rx', label="End Cycles")
            plt.xlabel("Time [s]")
            plt.ylabel("Knee Angle [rad]")
            plt.title("Knee Angle (Stila) with Start and Stop Cycles")
            plt.legend()
    
    
    #%% Truncate data based on cycles
    phi_dino = phi_dino[i_sf[0]:i_sf[-1]+1]
    torque_dino = torque_dino[i_sf[0]:i_sf[-1]+1]
    time_dino = time_dino[i_sf[0]:i_sf[-1]+1] - time_dino[i_sf[0]]  # Adjust time to start at 0
    
    if stila_data is not None:
        phi_stila = phi_stila[j_sf[0]:j_sf[-1]+1]
        torque_stila = torque_stila[j_sf[0]:j_sf[-1]+1]
        emg_stila = emg_stila[:,j_sf[0]:j_sf[-1]+1]
        time_stila = time_stila[j_sf[0]:j_sf[-1]+1] - time_stila[i_sf[0]]
    
    i_se = i_se-i_sf[0]
    j_se = j_se-j_sf[0]
    i_sf = i_sf-i_sf[0]
    j_sf = j_sf-j_sf[0]
    
    #%% Correct for the angle difference between dino and stila
    if stila_data is not None:
        phidev_stila = phi_stila - np.mean(phi_stila)                       # [rad] difference between current and avg. angle
        phidev_dino = phi_dino - np.mean(phi_dino)                          # [rad] difference between current and avg. angle
        amp_stila = (np.max(phidev_stila) - np.min(phidev_stila)) / 2       # [rad] amplitude
        amp_dino = (np.max(phidev_dino) - np.min(phidev_dino)) / 2          # [rad] amplitude
        phi_stila = phidev_stila * amp_dino / amp_stila + np.mean(phi_dino) # [rad] corrected angle
    
    #%%
    # Interpolate data to new knee angles
    phi_f = np.sort(phi_dino[i_sf])  # Full flexion
    phi_e = np.sort(phi_dino[i_se])  # Full extension
    
    phi_E = np.concatenate([phi_e[::-1][1:], np.linspace(phi_e[-1], phi_f[-1], 5004 - 2 * len(phi_f)), phi_f[::-1][1:]])
    phi_F = np.concatenate([phi_f[1:], np.linspace(phi_f[-1], phi_e[0], 5004 - 2 * len(phi_f)), phi_e[1:]])
    
    # Initialize empty lists for interpolated moments and time
    interpolated_phi = []
    interpolated_torques = []
    interpolated_times = []
    # Process each cycle
    for i in range(nr_cyc):
        time_f = time_dino[i_sf[i]:i_se[i]]
        time_e = time_dino[i_se[i]:i_sf[i+1]]
        phi_f = phi_dino[i_sf[i]:i_se[i]]
        phi_e = phi_dino[i_se[i]:i_sf[i+1]]
        M_f = torque_dino[i_sf[i]:i_se[i]]
        M_e = torque_dino[i_se[i]:i_sf[i+1]]
        
        # Sometimes we've two values of phi, so then average the y-value
        phi_f_unique = np.unique(phi_f)
        phi_e_unique = np.unique(phi_e)
        M_f_unique = np.array([M_f[phi_f == xi].mean() for xi in phi_f_unique])
        M_e_unique = np.array([M_e[phi_e == xi].mean() for xi in phi_e_unique])
        time_f_unique = np.array([time_f[phi_f == xi].mean() for xi in phi_f_unique])
        time_e_unique = np.array([time_e[phi_e == xi].mean() for xi in phi_e_unique])
        
        # Do the interpolation
        M_F = np.interp(phi_F,phi_f_unique,M_f_unique)
        M_E = np.interp(phi_E,phi_e_unique,M_e_unique)
        time_F = np.interp(phi_F,phi_f_unique,time_f_unique)
        time_E = np.interp(phi_E,phi_e_unique,time_e_unique)
        
        # Append to list
        interpolated_phi.append(np.concatenate([phi_F, phi_E[1:]]))
        interpolated_torques.append(np.concatenate([M_F, M_E[1:]]))
        interpolated_times.append(np.concatenate([time_F, time_E[1:]]))
                
    # Combine interpolated data
    Phi_int = np.array(interpolated_phi)
    Torque_Measured_int = np.array(interpolated_torques)
    Time_int = np.array(interpolated_times)
    
    #%% Compute compensated torque
    # Compensate for passive moment
    if pp == 2 and cond == 9 and trial == 1:
        ncycles = [2, 3, 4, 13, 14]
    elif pp == 4 and cond == 1 and trial == 1:
        ncycles = [3, 4, 12, 13, 14]
    else:
        ncycles = [2, 3, 4, 12, 13, 14]
    
    Torque_Passive_int = Torque_Measured_int[ncycles, :]  # Moment during passive cycles
    Torque_Compensated_int = Torque_Measured_int.copy()
    for i in range(nr_cyc):
        Torque_Compensated_int[i, :] = Torque_Measured_int[i,:] - np.nanmean(Torque_Passive_int, axis=0)
    
    #%% Calculate mechanical work per cycle
    Wm = np.zeros(nr_cyc) * np.nan
    Wc = np.zeros(nr_cyc) * np.nan
    
    for iCycle in range(nr_cyc):
        j = ~np.isnan(Torque_Measured_int[iCycle,:])
        Wm[iCycle] = -np.trapz(Torque_Measured_int[iCycle, j], Phi_int[iCycle,j])
        j = ~np.isnan(Torque_Compensated_int[iCycle,:])
        Wc[iCycle] = -np.trapz(Torque_Compensated_int[iCycle, j], Phi_int[iCycle,j])
    
    #%% Interpolate back to time
    Time_int[0,0] = 0
    Time_int[-1,-1] = time_dino[-1]
    Time = time_dino
    Phi = phi_dino
    f = interpolate.interp1d(Time_int.flatten(), Torque_Measured_int.flatten(), kind='linear', bounds_error=False)
    TorqueInt  = f(Time)
    Torque = torque_dino
    f = interpolate.interp1d(Time_int.flatten(), Torque_Compensated_int.flatten(), kind='linear', bounds_error=False)
    TorqueComp = f(Time)
        
    # Plot data
    if fig == 1:
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.plot(time_dino,torque_dino,Time_int,Torque_Measured_int,'--',Time,Torque,'-.')
        
        plt.subplot(121)
        plt.plot(TorqueInt-Torque)
    
    RMSE = (np.nansum((TorqueInt-torque_dino)**2)/len(Torque))**0.5
    print(RMSE)
    if RMSE>0.5:
        breakpoint()
    
    #%% Process EMG-signal - Moving Average
    # Compute EMG envelope
    emg_env = EnvEMG_MovAve(emg_stila,0.1,fs_stila)
    
    # Calculate MVC of quad and hams
    emg_quad1 = pd.read_csv(dataDir+f'/pp{pp:02d}_day{day}_mvc_quad_t1.csv', header=None).T.to_numpy()[[3]][0]
    emg_quad2 = pd.read_csv(dataDir+f'/pp{pp:02d}_day{day}_mvc_quad_t2.csv', header=None).T.to_numpy()[[3]][0]
    emg_hams1 = pd.read_csv(dataDir+f'/pp{pp:02d}_day{day}_mvc_hams_t1.csv', header=None).T.to_numpy()[[5]][0]
    emg_hams2 = pd.read_csv(dataDir+f'/pp{pp:02d}_day{day}_mvc_hams_t2.csv', header=None).T.to_numpy()[[5]][0]

    emg_env_quad1_mvc = EnvEMG_MovAve(emg_quad1,0.5,fs_stila) 
    emg_env_quad2_mvc = EnvEMG_MovAve(emg_quad2,0.5,fs_stila)
    mvc_quad = max((max(emg_env_quad1_mvc),max(emg_env_quad2_mvc)))
    
    emg_env_hams1_mvc = EnvEMG_MovAve(emg_hams1,0.5,fs_stila) 
    emg_env_hams2_mvc = EnvEMG_MovAve(emg_hams2,0.5,fs_stila)
    mvc_hams = max((max(emg_env_hams1_mvc),max(emg_env_hams2_mvc)))
        
    emg_norm = emg_env.copy()
    emg_norm[0] = emg_norm[0]/mvc_quad
    emg_norm[1] = emg_norm[1]/mvc_hams
        
    #%% Process EMG-signal - Butter
    # fc = 10 # cut-off frequency of lowpass filter
    # # Compute EMG envelope
    # emg_env = EnvEMG_Butter(emg_stila,fc,fs_stila)
    
    # Calculate MVC of quad and hams
    # emg_quad1 = pd.read_csv(dataDir+f'/pp{pp:02d}_day{day}_mvc_quad_t1.csv', header=None).T.to_numpy()[[3]][0]
    # emg_quad2 = pd.read_csv(dataDir+f'/pp{pp:02d}_day{day}_mvc_quad_t2.csv', header=None).T.to_numpy()[[3]][0]
    # emg_hams1 = pd.read_csv(dataDir+f'/pp{pp:02d}_day{day}_mvc_hams_t1.csv', header=None).T.to_numpy()[[5]][0]
    # emg_hams2 = pd.read_csv(dataDir+f'/pp{pp:02d}_day{day}_mvc_hams_t2.csv', header=None).T.to_numpy()[[5]][0]
    
    # fc = 10
    # emg_env_quad1_mvc = EnvEMG_Butter(emg_quad1,fc,fs_stila) 
    # emg_env_quad2_mvc = EnvEMG_Butter(emg_quad2,fc,fs_stila)
    # mvc_quad1 = MovAve(emg_env_quad1_mvc,0.5,fs_stila)
    # mvc_quad2 = MovAve(emg_env_quad2_mvc,0.5,fs_stila)
    # mvc_quad = max((max(mvc_quad1),max(mvc_quad2)))
    
    # emg_env_hams1_mvc = EnvEMG_Butter(emg_hams1,fc,fs_stila) 
    # emg_env_hams2_mvc = EnvEMG_Butter(emg_hams2,fc,fs_stila)
    # mvc_hams1 = MovAve(emg_env_hams1_mvc,0.5,fs_stila)
    # mvc_hams2 = MovAve(emg_env_hams2_mvc,0.5,fs_stila)
    # mvc_hams = max((max(mvc_hams1),max(mvc_hams2)))
        
    # emg_norm = emg_env.copy()
    # emg_norm[0] = emg_norm[0]/mvc_quad
    # emg_norm[1] = emg_norm[1]/mvc_hams
    
    #%% Cut EMG signal
    emg_down = signal.resample_poly(emg_stila, 1, fs_stila//fs_dino, axis=1, window=('kaiser', 5.0), padtype='constant', cval=None)
    emg_norm_down = signal.resample_poly(emg_norm, 1, fs_stila//fs_dino, axis=1, window=('kaiser', 5.0), padtype='constant', cval=None)
    
    emg_down = signal.resample_poly(emg_stila, len(phi_dino), len(emg_stila.T), axis=1, window=('kaiser', 5.0), padtype='constant', cval=None)
    emg_norm_down = signal.resample_poly(emg_norm, len(phi_dino), len(emg_stila.T), axis=1, window=('kaiser', 5.0), padtype='constant', cval=None)

    if abs(len(emg_down.T)-len(phi_dino))/len(phi_dino) > 0.005 or abs(len(emg_norm_down.T)-len(phi_dino))/len(phi_dino) > 0.005:
        breakpoint()
    EMG = emg_down[:,0:i_sf[-1]+1]
    EMGenv = emg_norm_down[:,0:i_sf[-1]+1]
    
    #%% Create variable indicating start flexion and extension
    # Create a new array with empty strings
    phase = np.array(['-' for _ in range(len(Time))], dtype=object)  # Assuming array size is at least 15
    
    # Set 'f' at the specified indices
    phase[i_sf] = "sf"
    phase[i_se] = "se"
    
    arr_list = phase.tolist()
    # Ensure 'sf' and 'se' are matched correctly in pairs
    for i in range(len(i_sf)):
        start_sf = i_sf[i]
        
        # Find the corresponding 'se' index after the current 'sf'
        if i < len(i_se):
            start_se = i_se[i]
            
            # Replace '-' between 'sf' and 'se' with 'f'
            for j in range(start_sf + 1, start_se):
                if arr_list[j] == '-':
                    arr_list[j] = 'f'
            
            # Replace '-' between 'se' and next 'sf' with 'e'
            if i + 1 < len(i_sf):  # Ensure we have another 'sf' after 'se'
                end_sf = i_sf[i + 1]
                for j in range(start_se + 1, end_sf):
                    if arr_list[j] == '-':
                        arr_list[j] = 'e'
    phase = np.array(arr_list)

    #%% Store again.
    # Create the output data dictionary
    data_dict = {
        "Time": Time,
        "KneeAngle": Phi,
        "Torque": Torque,
        "TorqueComp": TorqueComp,
        "EMG": EMG,
        "EMGenv": EMGenv,
        "Phase": phase
        }    
    return data_dict

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def xcov_np(x, y, max_lag=10):
    """
    Computes the cross-covariance between two signals `x` and `y` for a range of lags using numpy.
    
    Handles the case where the signals are of unequal lengths by zero-padding the shorter signal.
    
    Inputs:
        x (ndarray): First input signal.
        y (ndarray): Second input signal.
        max_lag (int): Maximum lag (both positive and negative).
        
    Outputs:
        c (ndarray): Cross-covariance values for each lag.
        lags (ndarray): Corresponding lag values.
    """
    # Zero-pad the shorter signal to match the length of the longer signal
    len_x, len_y = len(x), len(y)
    if len_x > len_y:
        y = np.pad(y, (0, len_x - len_y), 'constant')
    elif len_y > len_x:
        x = np.pad(x, (0, len_y - len_x), 'constant')

    # Normalize the signals by subtracting the mean
    x = x - np.mean(x)
    y = y - np.mean(y)
        
    # Compute the cross-correlation using numpy.correlate
    corr = np.correlate(x, y, mode='full')

    # Create lag array: from -max_lag to +max_lag
    lags = np.arange(-len(x) + 1, len(x))

    # Get the portion of the correlation that corresponds to the range of lags we want
    start = len(x) - max_lag - 1
    end = len(x) + max_lag - 1
    c = corr[start:end + 1]

    # Normalize to get the cross-covariance
    c /= np.std(x) * np.std(y) * len(x)  # Normalize by std and length of signals
    
    return c, lags[start:end + 1]