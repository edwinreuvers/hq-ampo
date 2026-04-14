import numpy as np
from scipy.interpolate import griddata

def D3(data, grid, **kwargs):
    N = kwargs.get('N', 100)
    method = kwargs.get('method', 'linear')
    
    x, y = grid
    
    # Make meshgrid
    X, Y = np.meshgrid(x, y)

    # Flatten meshgrid
    x_flat = X.ravel()
    y_flat = Y.ravel()
    data_flat = data.ravel()

    # Remove NaN values for interpolation
    mask = ~np.isnan(data_flat)
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]
    data_flat = data_flat[mask]

    # Define a finer grid for interpolation
    x_fine = np.linspace(x[0], x[-1], N)
    y_fine = np.linspace(y[0], y[-1], N)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    
    # Perform the interpolation
    data_fine = griddata((x_flat, y_flat), data_flat, (X_fine, Y_fine), method=method)
    grid_fine = (x_fine, y_fine)
    
    return data_fine, grid_fine

def D4(data, grid, **kwargs):
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