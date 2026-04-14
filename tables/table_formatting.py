import numpy as np

def format_str(x, n=0, fmt=None):
    """
    Return a formatted string.

    Parameters:
    -----------
    x : array-like or pandas Series
        Numeric values.
    n : int
        Number of digits for rounding. 
        - n > 0 : number of decimal places
        - n = 0 or n < 0 : integer / rounding to tens/hundreds
    fmt : str, optional
        Format string, e.g. '0.0f' or '0.2f'.
        If None, it is inferred from `n`.

    Returns
    -------
    str
        Formatted string like "123 ± 45" or "1.23 ± 0.45"
    """
    # Determine format string automatically
    if fmt is None:
        fmt = f"0.{n}f" if n > 0 else "0.0f"
    
    # Convert to array for unified handling
    x_arr = np.asarray(x, dtype=float)
    
    # Round
    x_arr = round(x, n)
    
    # Format elementwise
    formatter = np.vectorize(lambda v: f"{v:{fmt}}")

    return formatter(x_arr)

def format_mean_std(x, n=0, fmt=None, pm="±"):
    """
    Return mean ± standard deviation as a formatted string.

    Parameters:
    -----------
    x : array-like or pandas Series
        Numeric values.
    n : int
        Number of digits for rounding. 
        - n > 0 : number of decimal places
        - n = 0 or n < 0 : integer / rounding to tens/hundreds
    fmt : str, optional
        Format string, e.g. '0.0f' or '0.2f'.
        If None, it is inferred from `n`.
    pm: str, optional
        pm-sign: either ± or $\pm$

    Returns
    -------
    str
        Formatted string like "123 ± 45" or "1.23 ± 0.45"
    """
    # Determine format string automatically
    if fmt is None:
        fmt = f"0.{n}f" if n > 0 else "0.0f"
    
    x_mean = round(np.mean(x), n)
    x_std  = round(np.std(x), n)
    
    return f"{x_mean:{fmt}} {pm} {x_std:{fmt}}"