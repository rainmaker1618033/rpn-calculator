# ============================================================================
# FILE: rpn_calculator/matrices.py
# ============================================================================
"""Matrix operations"""

import numpy as np
from .errors import CalculatorError
from .utils import is_matrix, to_numpy_matrix, from_numpy_matrix

def register_operations(calc):
    """Register matrix operations"""
    return {
        # Creation and properties
        "MATRIX": lambda: _op_matrix(calc),
        "IDENTITY": lambda: _op_identity(calc),
        "MSIZE": lambda: _op_msize(calc),
        "TRANSPOSE": lambda: _op_transpose(calc),
        
        # Arithmetic
        "M+": lambda: _op_madd(calc),
        "M-": lambda: _op_msub(calc),
        "M*": lambda: _op_mmul(calc),
        "MSCALE": lambda: _op_mscale(calc),
        
        # Properties
        "DET": lambda: _op_det(calc),
        "TRACE": lambda: _op_trace(calc),
        "MINV": lambda: _op_minv(calc),
        "RANK": lambda: _op_rank(calc),
        
        # Advanced
        "EIGEN": lambda: _op_eigenvalues(calc),
        "EIGENVEC": lambda: _op_eigenvectors(calc),
        "RREF": lambda: _op_rref(calc),
        "MSOLVE": lambda: _op_msolve(calc),
    }

# Matrix Creation and Properties
def _op_matrix(calc):
    """Create matrix from rows on stack. Usage: row1 row2 ... n MATRIX"""
    if not calc.stack:
        raise CalculatorError("MATRIX needs number of rows")
    
    n = calc.pop()
    if not isinstance(n, int) or n < 1:
        calc.push(n)
        raise CalculatorError("MATRIX needs positive integer for number of rows")
    
    if len(calc.stack) < n:
        calc.push(n)
        raise CalculatorError(f"MATRIX needs {n} rows on stack")
    
    rows = []
    for _ in range(n):
        row = calc.pop()
        if not isinstance(row, (list, tuple)):
            for r in reversed(rows):
                calc.push(r)
            calc.push(row)
            calc.push(n)
            raise CalculatorError("MATRIX expects vectors (rows) on stack")
        rows.insert(0, list(row))
    
    if rows and not all(len(row) == len(rows[0]) for row in rows):
        for r in rows:
            calc.push(r)
        calc.push(n)
        raise CalculatorError("All matrix rows must have same length")
    
    calc.push(rows)
    print(f"Created {len(rows)}×{len(rows[0]) if rows else 0} matrix")

def _op_identity(calc):
    """Create n×n identity matrix"""
    if not calc.stack:
        raise CalculatorError("IDENTITY needs dimension n")
    
    n = calc.pop()
    if not isinstance(n, int) or n < 1:
        calc.push(n)
        raise CalculatorError("IDENTITY needs positive integer")
    
    identity = np.eye(n)
    calc.push(from_numpy_matrix(identity))
    print(f"Created {n}×{n} identity matrix")

def _op_msize(calc):
    """Get matrix dimensions (returns rows, cols)"""
    if not calc.stack:
        raise CalculatorError("MSIZE needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("MSIZE expects a matrix")
    
    rows = len(m)
    cols = len(m[0]) if rows > 0 else 0
    calc.push(rows)
    calc.push(cols)
    print(f"Matrix size: {rows}×{cols}")

def _op_transpose(calc):
    """Transpose matrix"""
    if not calc.stack:
        raise CalculatorError("TRANSPOSE needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("TRANSPOSE expects a matrix")
    
    try:
        arr = to_numpy_matrix(m)
        result = arr.T
        calc.push(from_numpy_matrix(result))
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in TRANSPOSE: {e}")

# Matrix Arithmetic
def _op_madd(calc):
    """Matrix addition"""
    if len(calc.stack) < 2:
        raise CalculatorError("M+ needs 2 matrices")
    
    b = calc.pop()
    a = calc.pop()
    
    if not (is_matrix(a) and is_matrix(b)):
        calc.push(a)
        calc.push(b)
        raise CalculatorError("M+ expects two matrices")
    
    try:
        arr_a = to_numpy_matrix(a)
        arr_b = to_numpy_matrix(b)
        
        if arr_a.shape != arr_b.shape:
            raise ValueError("Matrix dimensions must match")
        
        result = arr_a + arr_b
        calc.push(from_numpy_matrix(result))
    except Exception as e:
        calc.push(a)
        calc.push(b)
        raise CalculatorError(f"Error in M+: {e}")

def _op_msub(calc):
    """Matrix subtraction"""
    if len(calc.stack) < 2:
        raise CalculatorError("M- needs 2 matrices")
    
    b = calc.pop()
    a = calc.pop()
    
    if not (is_matrix(a) and is_matrix(b)):
        calc.push(a)
        calc.push(b)
        raise CalculatorError("M- expects two matrices")
    
    try:
        arr_a = to_numpy_matrix(a)
        arr_b = to_numpy_matrix(b)
        
        if arr_a.shape != arr_b.shape:
            raise ValueError("Matrix dimensions must match")
        
        result = arr_a - arr_b
        calc.push(from_numpy_matrix(result))
    except Exception as e:
        calc.push(a)
        calc.push(b)
        raise CalculatorError(f"Error in M-: {e}")

def _op_mmul(calc):
    """Matrix multiplication"""
    if len(calc.stack) < 2:
        raise CalculatorError("M* needs 2 matrices")
    
    b = calc.pop()
    a = calc.pop()
    
    is_a_matrix = is_matrix(a)
    is_b_matrix = is_matrix(b)
    
    if not is_a_matrix and not is_b_matrix:
        calc.push(a)
        calc.push(b)
        raise CalculatorError("M* needs at least one matrix")
    
    try:
        if is_a_matrix and not is_b_matrix:
            arr_a = to_numpy_matrix(a)
            result = b * arr_a
            calc.push(from_numpy_matrix(result))
        elif not is_a_matrix and is_b_matrix:
            arr_b = to_numpy_matrix(b)
            result = a * arr_b
            calc.push(from_numpy_matrix(result))
        else:
            arr_a = to_numpy_matrix(a)
            arr_b = to_numpy_matrix(b)
            
            if arr_a.shape[1] != arr_b.shape[0]:
                raise ValueError(f"Cannot multiply {arr_a.shape[0]}×{arr_a.shape[1]} "
                               f"by {arr_b.shape[0]}×{arr_b.shape[1]}")
            
            result = np.matmul(arr_a, arr_b)
            calc.push(from_numpy_matrix(result))
    except Exception as e:
        calc.push(a)
        calc.push(b)
        raise CalculatorError(f"Error in M*: {e}")

def _op_mscale(calc):
    """Scale matrix by scalar"""
    if len(calc.stack) < 2:
        raise CalculatorError("MSCALE needs matrix and scalar")
    
    scalar = calc.pop()
    m = calc.pop()
    
    if not is_matrix(m):
        calc.push(m)
        calc.push(scalar)
        raise CalculatorError("MSCALE expects matrix then scalar")
    
    try:
        arr = to_numpy_matrix(m)
        result = scalar * arr
        calc.push(from_numpy_matrix(result))
    except Exception as e:
        calc.push(m)
        calc.push(scalar)
        raise CalculatorError(f"Error in MSCALE: {e}")

# Matrix Properties
def _op_det(calc):
    """Determinant of matrix"""
    if not calc.stack:
        raise CalculatorError("DET needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("DET expects a matrix")
    
    try:
        arr = to_numpy_matrix(m)
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("DET requires square matrix")
        
        det = np.linalg.det(arr)
        if isinstance(det, complex) and abs(det.imag) < 1e-10:
            det = det.real
        calc.push(det)
        print(f"Determinant: {det}")
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in DET: {e}")

def _op_trace(calc):
    """Trace of matrix (sum of diagonal)"""
    if not calc.stack:
        raise CalculatorError("TRACE needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("TRACE expects a matrix")
    
    try:
        arr = to_numpy_matrix(m)
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("TRACE requires square matrix")
        
        trace = np.trace(arr)
        if isinstance(trace, complex) and abs(trace.imag) < 1e-10:
            trace = trace.real
        calc.push(trace)
        print(f"Trace: {trace}")
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in TRACE: {e}")

def _op_minv(calc):
    """Matrix inverse"""
    if not calc.stack:
        raise CalculatorError("MINV needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("MINV expects a matrix")
    
    try:
        arr = to_numpy_matrix(m)
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("MINV requires square matrix")
        
        det = np.linalg.det(arr)
        if abs(det) < 1e-10:
            raise ValueError("Matrix is singular (determinant ≈ 0)")
        
        inv = np.linalg.inv(arr)
        calc.push(from_numpy_matrix(inv))
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in MINV: {e}")

def _op_rank(calc):
    """Matrix rank"""
    if not calc.stack:
        raise CalculatorError("RANK needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("RANK expects a matrix")
    
    try:
        arr = to_numpy_matrix(m)
        rank = np.linalg.matrix_rank(arr)
        calc.push(rank)
        print(f"Rank: {rank}")
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in RANK: {e}")

# Advanced Operations
def _op_eigenvalues(calc):
    """Compute eigenvalues of matrix"""
    if not calc.stack:
        raise CalculatorError("EIGEN needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("EIGEN expects a matrix")
    
    try:
        arr = to_numpy_matrix(m)
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("EIGEN requires square matrix")
        
        eigenvalues = np.linalg.eigvals(arr)
        result = []
        for val in eigenvalues:
            if isinstance(val, complex) and abs(val.imag) < 1e-10:
                val = val.real
            if isinstance(val, float) and abs(val - round(val)) < 1e-10:
                val = int(round(val))
            result.append(val)
        
        calc.push(result)
        print(f"Eigenvalues: {result}")
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in EIGEN: {e}")

def _op_eigenvectors(calc):
    """Compute eigenvalues and eigenvectors"""
    if not calc.stack:
        raise CalculatorError("EIGENVEC needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("EIGENVEC expects a matrix")
    
    try:
        arr = to_numpy_matrix(m)
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("EIGENVEC requires square matrix")
        
        eigenvalues, eigenvectors = np.linalg.eig(arr)
        
        evals = []
        for val in eigenvalues:
            if isinstance(val, complex) and abs(val.imag) < 1e-10:
                val = val.real
            if isinstance(val, float) and abs(val - round(val)) < 1e-10:
                val = int(round(val))
            evals.append(val)
        
        calc.push(evals)
        calc.push(from_numpy_matrix(eigenvectors))
        print(f"Eigenvalues: {evals}")
        print("Eigenvectors pushed as columns of matrix")
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in EIGENVEC: {e}")

def _op_rref(calc):
    """Reduced row echelon form"""
    if not calc.stack:
        raise CalculatorError("RREF needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("RREF expects a matrix")
    
    try:
        arr = to_numpy_matrix(m).astype(float)
        rows, cols = arr.shape
        
        current_row = 0
        for col in range(cols):
            if current_row >= rows:
                break
            
            pivot_row = current_row
            for row in range(current_row + 1, rows):
                if abs(arr[row, col]) > abs(arr[pivot_row, col]):
                    pivot_row = row
            
            if abs(arr[pivot_row, col]) < 1e-10:
                continue
            
            if pivot_row != current_row:
                arr[[current_row, pivot_row]] = arr[[pivot_row, current_row]]
            
            arr[current_row] = arr[current_row] / arr[current_row, col]
            
            for row in range(rows):
                if row != current_row:
                    arr[row] = arr[row] - arr[row, col] * arr[current_row]
            
            current_row += 1
        
        arr[np.abs(arr) < 1e-10] = 0
        calc.push(from_numpy_matrix(arr))
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in RREF: {e}")

def _op_msolve(calc):
    """Solve linear system Ax = b"""
    if len(calc.stack) < 2:
        raise CalculatorError("MSOLVE needs matrix A and vector b")
    
    b = calc.pop()
    a = calc.pop()
    
    if not is_matrix(a):
        calc.push(a)
        calc.push(b)
        raise CalculatorError("MSOLVE expects matrix A")
    
    if not isinstance(b, (list, tuple)):
        calc.push(a)
        calc.push(b)
        raise CalculatorError("MSOLVE expects vector b")
    
    try:
        arr_a = to_numpy_matrix(a)
        arr_b = np.array(b, dtype=complex if any(isinstance(v, complex) for v in b) else float)
        
        if arr_a.shape[0] != arr_a.shape[1]:
            raise ValueError("MSOLVE requires square matrix A")
        
        if len(arr_b) != arr_a.shape[0]:
            raise ValueError("Vector b length must match matrix A rows")
        
        x = np.linalg.solve(arr_a, arr_b)
        
        result = []
        for val in x:
            if isinstance(val, complex) and abs(val.imag) < 1e-10:
                val = val.real
            if isinstance(val, float) and abs(val - round(val)) < 1e-10:
                val = int(round(val))
            result.append(val)
        
        calc.push(result)
        print(f"Solution: {result}")
    except Exception as e:
        calc.push(a)
        calc.push(b)
        raise CalculatorError(f"Error in MSOLVE: {e}")