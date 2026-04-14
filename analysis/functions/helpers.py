"""
This module contains some helper functions to analyse the data. 

Functions
---------
segments_to_list(x, phase, part="cycle", cycle_indices=None)
    Extract segments of a signal corresponding to specific phases and return
    them as a list of arrays (one per cycle).

segments_to_array(x, phase, part="cycle", cycle_indices=None)
    Extract segments and store them in a 2D NumPy array, padding shorter
    segments with NaNs.

data_cut(x_list, phi_list, bounds)
    Cut segmented data based on value bounds (e.g., angle thresholds).

max_and_indices(arr, collapse_axes)
    Compute maximum values along specified axes and return both the maxima
    and their corresponding indices.
"""

import numpy as np

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def segments_to_list(x, phase, part="cycle", cycle_indices=None):
    """
    Extract segments of a signal corresponding to specific phases and store it
    in a list.

    Each list item corresponds to one segment.

    Parameters
    ----------
    x : np.ndarray
        1D array containing the signal to segment (e.g., joint angle or torque).
    phase : list or array-like of str
        Phase labels corresponding to each sample in `signal`.
        Must contain markers for:
        - 'se' : start of extension
        - 'sf' : start of flexion
    part : {'cycle', 'c', 'flexion', 'f', 'extension', 'e'}, optional
        Which part of the cycle to extract:
        - 'cycle'     : full cycle (se → se)
        - 'flexion'   : flexion phase (sf → se)
        - 'extension' : extension phase (se → sf)
    cycle_indices : array-like of int (optional)
        Indices of cycles to extract. Each cycle is defined from one 'se' 
        to the next 'se'. If not defined then all cycles are computed

    Returns
    -------
    segments : list of np.ndarray
        List of extracted segments for each requested cycle.

    Notes
    -----
    - A full cycle is defined from one 'se' (start extension) to the next.
    - Flexion is defined from 'sf' to the next 'se'.
    - Extension is defined from 'se' to the next 'sf'.
    """

    # Find phase transition indices
    i_sf = [i for i, p in enumerate(phase) if "sf" in p]
    i_se = [i for i, p in enumerate(phase) if "se" in p]
    
    if len(i_se) < 2:
        raise ValueError("Not enough 'se' markers to define cycles.")
    if len(i_sf) < 1:
        raise ValueError("No 'sf' markers found.")
    if cycle_indices is None:
        cycle_indices = np.arange(0,len(i_se)-1)
    
    # Extract segments
    y = []
    for i in cycle_indices:
        # Safety check for indexing
        if i + 1 >= len(i_se):
            raise IndexError(f"Cycle index {i} out of range.")

        part = part.lower()
        if part in ["cycle", "cyc", "c"]:
            idx = slice(i_se[i], i_se[i + 1] + 1)
        elif part in ["flexion", "flex", "flx", "f"]:
            idx = slice(i_sf[i + 1], i_se[i + 1] + 1)
        elif part in ["extension", "ext", "e"]:
            idx = slice(i_se[i], i_sf[i + 1] + 1)
        else:
            raise ValueError(f"Unknown part: {part}")

        y.append(x[idx])

    return y

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def segments_to_array(x, phase, part='cycle', cycle_indices=None):
    """
    Extract segments of a signal corresponding to specific phases and store it
    in a 2D array.

    Each column corresponds to one segment, and rows are padded with NaNs
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
    cycle_indices : array-like of int (optional)
        Indices of cycles to extract. Each cycle is defined from one 'se' 
        to the next 'se'. If not defined then all cycles are computed
    
    Returns
    -------
    segments : np.ndarray
        2D array of shape (max_cycle_length, n_cycles), where each column is a cycle.
        Shorter cycles are padded with NaNs.

    Notes
    -----
    - A full cycle is defined from one 'se' (start extension) to the next.
    - Flexion is defined from 'sf' to the next 'se'.
    - Extension is defined from 'se' to the next 'sf'.
    
    """
    # Extract each cycle as a list of arrays
    xList = segments_to_list(x, phase, part, cycle_indices)

    # Determine the maximum length of all cycles
    nRow = max(len(arr) for arr in xList)
    nCol = len(xList)
    segments = np.full((nRow, nCol), np.nan)

    # Fill each column with the corresponding cycle data
    for col_idx, cycle_data in enumerate(xList):
        segments[:len(cycle_data), col_idx] = cycle_data

    return segments

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def data_cut(x_list, phi_list, bounds):
    """
    Select segments of data within specified angle bounds.

    Parameters
    ----------
    x_list : list of np.ndarray
        List of data arrays (e.g., torque per cycle).
    phi_list : list of np.ndarray
        List of corresponding angle arrays (e.g., joint angle per cycle).
        Must have the same structure as `x_list`.
    bounds : tuple or list of float
        Lower and upper bounds for angle selection: (min_angle, max_angle).

    Returns
    -------
    x_cut : list of np.ndarray
        Cutted data arrays where `phi` is within bounds.
    phi_cut : list of np.ndarray
        Cutted angle arrays within bounds.

    Notes
    -----
    - Selection is inclusive: bounds[0] <= phi <= bounds[1]
    - Assumes each element in `x_list` corresponds to the same-length
      element in `phi_list`.
    """

    x_cut = []
    phi_cut = []

    lower, upper = bounds

    for x_cycle, phi_cycle in zip(x_list, phi_list):
        # Boolean mask for selecting values within bounds
        mask = (phi_cycle >= lower) & (phi_cycle <= upper)

        x_cut.append(x_cycle[mask])
        phi_cut.append(phi_cycle[mask])

    return x_cut, phi_cut

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def max_and_indices(arr, collapse_axes):
    """
    Compute max values along specified axes of an N-D array, and return
    the indices along the collapsed axes corresponding to the maxima.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array.
    collapse_axes : tuple or list of ints
        Axes along which to compute the max.
        
    Returns
    -------
    max_vals : np.ndarray
        Max values along the collapsed axes.
    indices : tuple of np.ndarray
        Indices along each collapsed axis corresponding to the maxima.
        Each array has the same shape as max_vals.
    """
    # All axes
    axes = tuple(range(arr.ndim))
    
    # Remaining axes = those not collapsed
    remain_axes = [ax for ax in axes if ax not in collapse_axes]
    
    # Transpose so remaining axes come first, collapsed axes come last
    transpose_order = remain_axes + list(collapse_axes)
    arr_t = np.transpose(arr, transpose_order)
    
    # Flatten collapsed axes into one
    shape_remain = [arr.shape[ax] for ax in remain_axes]
    shape_flat = (-1, np.prod([arr.shape[ax] for ax in collapse_axes]))
    arr_flat = arr_t.reshape(shape_flat)
    
    # Max along flattened collapsed axes
    idx_flat = np.nanargmax(arr_flat, axis=1)
    max_vals = np.nanmax(arr_flat, axis=1)
    
    # Unravel indices for collapsed axes
    collapsed_shape = [arr.shape[ax] for ax in collapse_axes]
    unraveled = np.array(np.unravel_index(idx_flat, collapsed_shape))
    
    # Reshape max_vals and unraveled indices to match remaining axes
    max_vals = max_vals.reshape(shape_remain)
    indices = tuple(u.reshape(shape_remain) for u in unraveled)
    
    return max_vals, indices
