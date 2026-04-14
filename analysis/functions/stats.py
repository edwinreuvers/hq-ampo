# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:13:47 2022

@author: Edwin
"""

import numpy as np

def ceil(a, precision=0):
    """
    Ceil a number or array to the given decimal precision.
    
    Inputs:
        a : float or array-like
        precision : int, number of decimal places
    
    Outputs:
        float or array-like with ceiling applied
    """
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def floor(a, precision=0):
    """
    Floor a number or array to the given decimal precision.
    
    Inputs:
        a : float or array-like
        precision : int, number of decimal places
    
    Outputs:
        float or array-like with floor applied
    """
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

def rmse(x, y):
    """
    Compute the Root Mean Square Error between two arrays.
    
    Inputs:
        x, y : array-like
    
    Outputs:
        float, RMSE
    """
    mse = np.mean((x - y) ** 2)
    return float(np.sqrt(mse))

def pdiff(x,y):
    """
    Compute the percentage difference between with x w.r.t. y.
    
    Inputs:
        x, y : np.array of pd.dataframe of similar size
    
    Outputs:
        float, percentage difference
    """
    return ((x-y)/y)*100

def zscore(x, axis=-1):
    """
    Compute the z-score normalization along a given axis.

    Inputs:
    ----------
    x : np.ndarray
        Input array.
    axis : int, optional
        Axis along which to compute the mean and std. Default is -1.

    Outputs
    -------
    np.ndarray
        Z-score normalized array.
    """
    m = np.mean(x, axis=axis, keepdims=True)
    s = np.std(x, axis=axis, keepdims=True)
    return (x - m) / s

def str_round(value: float, n: int) -> str:
    """
    Round a number to n significant digits and return a string representation.
    
    Inputs:
        value : float
        n : int, number of significant digits
    
    Outputs:
        str, rounded value with exactly n significant digits
    """
    
    if value == 0:
        return '0'
    
    # Determine the number of digits before the decimal point
    num_digits = int(np.floor(np.log10(abs(value)))) + 1
    
    # Calculate the decimal places to round
    decimal_places = max(n - num_digits, 0)
    
    rounded_value = round(value, decimal_places)
    
    # Format as string with fixed decimal places if needed
    if decimal_places > 0:
        return f"{rounded_value:.{decimal_places}f}"
    else:
        return str(int(rounded_value))
    

