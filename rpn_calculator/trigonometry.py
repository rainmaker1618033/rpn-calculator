# ============================================================================
# FILE: rpn_calculator/trigonometry.py
# ============================================================================
"""Trigonometric operations"""

import math
import cmath
from typing import Callable, Optional
from .errors import CalculatorError

def register_operations(calc):
    """Register trigonometric operations"""
    return {
        "SIN": lambda: _unary_trig(calc, math.sin, cmath.sin, "SIN"),
        "COS": lambda: _unary_trig(calc, math.cos, cmath.cos, "COS"),
        "TAN": lambda: _unary_trig(calc, math.tan, cmath.tan, "TAN"),
        "ASIN": lambda: _unary_inverse_trig(calc, math.asin, cmath.asin, "ASIN", -1, 1),
        "ACOS": lambda: _unary_inverse_trig(calc, math.acos, cmath.acos, "ACOS", -1, 1),
        "ATAN": lambda: _unary_inverse_trig(calc, math.atan, cmath.atan, "ATAN"),
    }

def _unary_trig(calc, real_func: Callable, complex_func: Callable, name: str):
    """Execute a unary trigonometric operation"""
    if not calc.stack:
        raise CalculatorError(f"Stack empty before '{name}'")
    
    x = calc.pop()
    try:
        is_complex = isinstance(x, complex)
        if is_complex:
            result = complex_func(x)
        else:
            angle = math.radians(x) if calc.state.degrees else x
            result = real_func(angle)
        calc.push(result)
    except Exception as e:
        calc.push(x)
        raise CalculatorError(f"Error in {name}: {e}")

def _unary_inverse_trig(calc, real_func: Callable, complex_func: Callable, 
                       name: str, min_val: Optional[float] = None, 
                       max_val: Optional[float] = None):
    """Execute a unary inverse trigonometric operation"""
    if not calc.stack:
        raise CalculatorError(f"Stack empty before '{name}'")
    
    x = calc.pop()
    try:
        is_complex = isinstance(x, complex)
        out_of_range = (min_val is not None and max_val is not None and 
                      not (min_val <= x <= max_val))
        
        if is_complex or out_of_range:
            result = complex_func(x)
        else:
            result = real_func(x)
            if not isinstance(result, complex):
                result = math.degrees(result) if calc.state.degrees else result
        
        calc.push(result)
    except Exception as e:
        calc.push(x)
        raise CalculatorError(f"Error in {name}: {e}")