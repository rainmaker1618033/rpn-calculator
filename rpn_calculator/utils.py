# ============================================================================
# FILE: rpn_calculator/utils.py
# ============================================================================
"""Utility functions used across modules"""

import numpy as np

def is_vector(x) -> bool:
    """Check if x is a vector (1D list/tuple)"""
    if isinstance(x, np.ndarray):
        return x.ndim == 1
    return isinstance(x, (list, tuple)) and x and not isinstance(x[0], (list, tuple))

def is_matrix(x) -> bool:
    """Check if x is a valid matrix (2D list/array)"""
    if not isinstance(x, (list, tuple, np.ndarray)):
        return False
    if isinstance(x, np.ndarray):
        return x.ndim == 2
    if not x:  # empty list
        return False
    # Check if it's a list of lists
    if not isinstance(x[0], (list, tuple)):
        return False
    # Check all rows have same length
    row_len = len(x[0])
    return all(isinstance(row, (list, tuple)) and len(row) == row_len for row in x)

def to_numpy_matrix(x):
    """Convert to numpy matrix, handling complex numbers"""
    if isinstance(x, np.ndarray):
        return x
    return np.array(x, dtype=complex if any(isinstance(val, complex) 
                    for row in x for val in row) else float)

def from_numpy_matrix(arr):
    """Convert numpy array back to list format"""
    result = []
    for row in arr:
        new_row = []
        for val in row:
            if isinstance(val, complex):
                if abs(val.imag) < 1e-10:
                    val = val.real
            if isinstance(val, float) and abs(val - round(val)) < 1e-10:
                val = int(round(val))
            new_row.append(val)
        result.append(new_row)
    return result