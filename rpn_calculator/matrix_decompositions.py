# ============================================================================
# FILE: rpn_calculator/matrix_decompositions.py
# ============================================================================
"""Matrix decomposition operations (LU, QR, SVD, etc.)"""

import numpy as np
from .errors import CalculatorError
from .utils import is_matrix, to_numpy_matrix, from_numpy_matrix

def register_operations(calc):
    """Register matrix decomposition operations"""
    return {
        "LU": lambda: _op_lu(calc),
        "QR": lambda: _op_qr(calc),
        "SVD": lambda: _op_svd(calc),
        "CHOLESKY": lambda: _op_cholesky(calc),
        "SCHUR": lambda: _op_schur(calc),
        "HESSENBERG": lambda: _op_hessenberg(calc),
        "COND": lambda: _op_condition(calc),
        "NORM": lambda: _op_norm(calc),
    }

def _op_lu(calc):
    """LU decomposition with partial pivoting: A = PLU"""
    if not calc.stack:
        raise CalculatorError("LU needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("LU expects a matrix")
    
    try:
        arr = to_numpy_matrix(m)
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("LU requires square matrix")
        
        from scipy.linalg import lu
        P, L, U = lu(arr)
        
        calc.push(from_numpy_matrix(P))
        calc.push(from_numpy_matrix(L))
        calc.push(from_numpy_matrix(U))
        print("LU decomposition complete: P, L, U pushed to stack")
        print("Verify: P·L·U = A")
    except ImportError:
        calc.push(m)
        raise CalculatorError("LU requires scipy library (pip install scipy)")
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in LU: {e}")

def _op_qr(calc):
    """QR decomposition: A = QR"""
    if not calc.stack:
        raise CalculatorError("QR needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("QR expects a matrix")
    
    try:
        arr = to_numpy_matrix(m)
        Q, R = np.linalg.qr(arr)
        
        calc.push(from_numpy_matrix(Q))
        calc.push(from_numpy_matrix(R))
        print("QR decomposition complete: Q, R pushed to stack")
        print("Q is orthogonal, R is upper triangular")
        print("Verify: Q·R = A")
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in QR: {e}")

def _op_svd(calc):
    """Singular Value Decomposition: A = UΣV*"""
    if not calc.stack:
        raise CalculatorError("SVD needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("SVD expects a matrix")
    
    try:
        arr = to_numpy_matrix(m)
        U, S, Vh = np.linalg.svd(arr, full_matrices=True)
        
        rows, cols = arr.shape
        Sigma = np.zeros((rows, cols), dtype=arr.dtype)
        min_dim = min(rows, cols)
        Sigma[:min_dim, :min_dim] = np.diag(S)
        
        calc.push(from_numpy_matrix(U))
        calc.push(from_numpy_matrix(Sigma))
        calc.push(from_numpy_matrix(Vh))
        print("SVD complete: U, Σ, V* pushed to stack")
        print("U and V* are orthogonal/unitary, Σ is diagonal")
        print("Verify: U·Σ·V* = A")
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in SVD: {e}")

def _op_cholesky(calc):
    """Cholesky decomposition: A = L·L*"""
    if not calc.stack:
        raise CalculatorError("CHOLESKY needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("CHOLESKY expects a matrix")
    
    try:
        arr = to_numpy_matrix(m)
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("CHOLESKY requires square matrix")
        
        L = np.linalg.cholesky(arr)
        calc.push(from_numpy_matrix(L))
        print("Cholesky decomposition complete: L pushed to stack")
        print("L is lower triangular")
        print("Verify: L·L* = A")
    except np.linalg.LinAlgError:
        calc.push(m)
        raise CalculatorError("CHOLESKY requires positive definite matrix")
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in CHOLESKY: {e}")

def _op_schur(calc):
    """Schur decomposition: A = Q·T·Q*"""
    if not calc.stack:
        raise CalculatorError("SCHUR needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("SCHUR expects a matrix")
    
    try:
        arr = to_numpy_matrix(m)
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("SCHUR requires square matrix")
        
        from scipy.linalg import schur
        T, Q = schur(arr)
        
        calc.push(from_numpy_matrix(Q))
        calc.push(from_numpy_matrix(T))
        print("Schur decomposition complete: Q, T pushed to stack")
        print("Q is unitary, T is upper triangular")
        print("Verify: Q·T·Q* = A")
    except ImportError:
        calc.push(m)
        raise CalculatorError("SCHUR requires scipy library (pip install scipy)")
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in SCHUR: {e}")

def _op_hessenberg(calc):
    """Hessenberg decomposition: A = Q·H·Q*"""
    if not calc.stack:
        raise CalculatorError("HESSENBERG needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("HESSENBERG expects a matrix")
    
    try:
        arr = to_numpy_matrix(m)
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("HESSENBERG requires square matrix")
        
        from scipy.linalg import hessenberg
        H, Q = hessenberg(arr, calc_q=True)
        
        calc.push(from_numpy_matrix(Q))
        calc.push(from_numpy_matrix(H))
        print("Hessenberg decomposition complete: Q, H pushed to stack")
        print("Q is orthogonal, H is upper Hessenberg")
        print("Verify: Q·H·Q* = A")
    except ImportError:
        calc.push(m)
        raise CalculatorError("HESSENBERG requires scipy library (pip install scipy)")
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in HESSENBERG: {e}")

def _op_condition(calc):
    """Condition number of matrix"""
    if not calc.stack:
        raise CalculatorError("COND needs a matrix")
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("COND expects a matrix")
    
    try:
        arr = to_numpy_matrix(m)
        cond = np.linalg.cond(arr)
        
        if np.isinf(cond):
            print("Condition number: ∞ (singular matrix)")
        else:
            print(f"Condition number: {cond:.6e}")
            if cond > 1e10:
                print("Warning: Matrix is ill-conditioned")
        
        calc.push(float(cond))
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in COND: {e}")

def _op_norm(calc):
    """Matrix norm (Frobenius norm by default)"""
    if not calc.stack:
        raise CalculatorError("NORM needs a matrix")
    
    norm_type = 'fro'
    if len(calc.stack) >= 2 and isinstance(calc.stack[-1], str):
        norm_str = calc.pop().lower()
        if norm_str in ['1', '2', 'inf', 'fro', 'nuc']:
            norm_type = norm_str if norm_str != 'inf' else np.inf
        else:
            calc.push(norm_str)
    
    m = calc.pop()
    if not is_matrix(m):
        calc.push(m)
        raise CalculatorError("NORM expects a matrix")
    
    try:
        arr = to_numpy_matrix(m)
        norm_val = np.linalg.norm(arr, ord=norm_type)
        
        if isinstance(norm_val, complex) and abs(norm_val.imag) < 1e-10:
            norm_val = norm_val.real
        
        calc.push(float(norm_val))
        print(f"Matrix norm ({norm_type}): {norm_val}")
    except Exception as e:
        calc.push(m)
        raise CalculatorError(f"Error in NORM: {e}")