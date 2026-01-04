# ============================================================================
# FILE: rpn_calculator/statistics.py
# ============================================================================
"""Statistics operations"""

import math
from .errors import CalculatorError

def register_operations(calc):
    """Register statistics operations"""
    return {
        "COMB": lambda: _op_comb(calc),
        "PERM": lambda: _op_perm(calc),
        "STDV": lambda: _op_stdv(calc),
    }

# Statistics Operations
def _op_comb(calc):
    if len(calc.stack) < 2:
        raise CalculatorError("COMB requires n and k")
    
    k = calc.pop()
    n = calc.pop()
    
    try:
        # Try to interpret as integers
        if isinstance(n, float) and n.is_integer():
            n = int(n)
        if isinstance(k, float) and k.is_integer():
            k = int(k)
        
        if not (isinstance(n, int) and isinstance(k, int)):
            raise ValueError("COMB requires integer operands")
        
        if k < 0 or n < 0 or k > n:
            raise ValueError("Invalid domain for COMB")
        
        result = math.comb(n, k)
        calc.push(result)
        
    except Exception as e:
        calc.push(n)
        calc.push(k)
        raise CalculatorError(f"Error in COMB: {e}")

def _op_perm(calc):
    if len(calc.stack) < 2:
        raise CalculatorError("PERM requires n and k")
    
    k = calc.pop()
    n = calc.pop()
    
    try:
        if isinstance(n, float) and n.is_integer():
            n = int(n)
        if isinstance(k, float) and k.is_integer():
            k = int(k)
        
        if not (isinstance(n, int) and isinstance(k, int)):
            raise ValueError("PERM requires integer operands")
        
        if k < 0 or n < 0 or k > n:
            raise ValueError("Invalid domain for PERM")
        
        result = math.perm(n, k)
        calc.push(result)
        
    except Exception as e:
        calc.push(n)
        calc.push(k)
        raise CalculatorError(f"Error in PERM: {e}")

def _op_stdv(calc):
    if not calc.stack:
        raise CalculatorError("STDV requires a vector")
    
    v = calc.pop()
    
    if not isinstance(v, (list, tuple)):
        calc.push(v)
        raise CalculatorError("STDV expects a vector")
    
    try:
        if len(v) == 0:
            raise ValueError("Empty vector")
        
        mean = sum(v) / len(v)
        var = sum((x - mean)**2 for x in v) / len(v)
        std = var**0.5
        
        calc.push(std)
        
    except Exception as e:
        calc.push(v)
        raise CalculatorError(f"Error in STDV: {e}")