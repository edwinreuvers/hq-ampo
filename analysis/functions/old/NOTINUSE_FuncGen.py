import numpy as np
from scipy.interpolate import griddata



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def data2matrix(x, Phase, part='cycle'):
    """
    Convert segmented cycle data into a 2D matrix.

    Each column corresponds to one cycle, and rows are padded with NaNs
    if cycles have different lengths.

    Parameters
    ----------
    x : array-like
        The data to be segmented, e.g., time, angle, or torque.
    Phase : list of str
        Phase labels for each sample (e.g., 'sf' for start flexion, 'se' for start extension).
    part : str, optional
        Part of the cycle to extract. Options are:
        - 'cycle' (default): full cycle from start extension to start extension.
        - 'flexion', 'flx': flexion phase.
        - 'extension', 'ext': extension phase.

    Returns
    -------
    np.ndarray
        2D array of shape (max_cycle_length, n_cycles), where each column is a cycle.
        Shorter cycles are padded with NaNs.
    """
    # Extract each cycle as a list of arrays
    n_cycles = 14
    xList = data2list(x, Phase, np.arange(n_cycles), part)

    # Determine the maximum length of all cycles
    nRow = max(len(arr) for arr in xList)
    y = np.full((nRow, n_cycles), np.nan)

    # Fill each column with the corresponding cycle data
    for col_idx, cycle_data in enumerate(xList):
        y[:len(cycle_data), col_idx] = cycle_data

    return y

def interpol3D(x, y, z, N=100):
    # Interpolate data
    X, Y = np.meshgrid(x, y)

    # Flatten meshgrid
    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = z.ravel()

    # Remove NaN values for interpolation
    mask = ~np.isnan(z_flat)
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]
    z_flat = z_flat[mask]

    # Define a finer grid for interpolation
    x_fine = np.linspace(x[0], x[-1], N)
    y_fine = np.linspace(y[0], y[-1], N)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    
    # Perform the interpolation
    z_fine = griddata((x_flat, y_flat), z_flat, (X_fine, Y_fine), method='cubic')

    return x_fine, y_fine, z_fine

def interpol4D(data, grid, **kwargs):
    N = kwargs.get('N', 100)
    method = kwargs.get('method', 'linear')
    
    x, y, z = grid
    
    # Make meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten meshgrid
    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = Z.ravel()
    data_flat = data.ravel()
        
    # Remove NaN values for interpolation
    mask = ~np.isnan(data_flat)
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]
    z_flat = z_flat[mask]
    data_flat = data_flat[mask]    
    
    # Define a finer grid for interpolation
    x_fine = np.linspace(x[0], x[-1], N)
    y_fine = np.linspace(y[0], y[-1], N+1)
    z_fine = np.linspace(z[0], z[-1], N+2)
    X_fine, Y_fine, Z_fine = np.meshgrid(x_fine, y_fine, z_fine, indexing='ij')
    
    
    # Perform the interpolation using griddata
    interpolated_values = griddata(
        points=np.vstack([x_flat, y_flat, z_flat]).T,
        values=data_flat,
        xi=np.vstack([X_fine.ravel(), Y_fine.ravel(), Z_fine.ravel()]).T,
        method=method  # or 'nearest'
    )
    
    data_fine = interpolated_values.reshape((N, N+1, N+2))
    grid_fine = (x_fine, y_fine, z_fine)
    return data_fine, grid_fine


def roundValue(value,n):
    if np.isnan(value):
        return '-'
    
    # Convert value to string to count digits
    value_str = str(value)
    
    # Find the number of digits before the decimal point
    if '.' in value_str:
        integer_part = value_str.split('.')[0]
    else:
        integer_part = value_str  # No decimal part

    num_digits = len(integer_part)
    
    # Determine how many decimal places to round to
    if num_digits < n:
        decimal_places = n - num_digits  # Round to n - number of digits before decimal
        rounded_value = round(value, decimal_places)
        value = f"{rounded_value:.{decimal_places}f}"
        if int(value.split('.')[0])>num_digits:
            return value[0:4]
        else:
            return value
    else:
        rounded_value = round(value, - (num_digits - n))  # Round to significant digits
        return str(int(rounded_value))  # Convert to int if there are enough digits