"""Logarithmic and exponential operations"""

import math
import cmath
from .errors import CalculatorError

def register_operations(calc):
    """Register logarithmic and exponential operations"""
    return {
        "LOG": lambda: _unary_math(calc, math.log10, cmath.log10, "LOG"),
        "LOG2": lambda: _unary_math(calc, math.log2, lambda x: cmath.log(x, 2), "LOG2"),
        "LN": lambda: _unary_math(calc, math.log, cmath.log, "LN"),
        "EXP": lambda: _unary_math(calc, math.exp, cmath.exp, "EXP"),
        "SQRT": lambda: _unary_math(calc, math.sqrt, cmath.sqrt, "SQRT"),
        "1/X": lambda: _op_inverse(calc),
        "INV": lambda: _op_inverse(calc),
    }

def _unary_math(calc, real_func, complex_func, name: str):
    if not calc.stack:
        raise CalculatorError(f"Stack empty before '{name}'")
    
    x = calc.pop()
    try:
        is_complex = isinstance(x, complex)
        needs_complex = (name in {"LOG", "LOG2", "LN"} and x <= 0) or \
                      (name == "SQRT" and x < 0)
        
        if is_complex or needs_complex:
            result = complex_func(x)
        else:
            result = real_func(x)
        
        calc.push(result)
    except Exception as e:
        calc.push(x)
        raise CalculatorError(f"Error in {name}: {e}")

def _op_inverse(calc):
    if not calc.stack:
        raise CalculatorError("Stack empty before '1/X'")
    
    x = calc.pop()
    try:
        if x == 0:
            raise ZeroDivisionError("Cannot invert zero")
        calc.push(1 / x)
    except Exception as e:
        calc.push(x)
        raise CalculatorError(f"Error in 1/X: {e}")