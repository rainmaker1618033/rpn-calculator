# ============================================================================
# FILE: rpn_calculator/integer_ops.py
# ============================================================================
"""Integer operations (GCD, LCM, FRAC)"""

import math
from .errors import CalculatorError

def register_operations(calc):
    """Register integer operations"""
    return {
        "GCD": lambda: _op_gcd(calc),
        "LCM": lambda: _op_lcm(calc),
        "FRAC": lambda: _op_frac(calc),
    }

# Integer Operations
def _op_gcd(calc):
    if len(calc.stack) < 2:
        raise CalculatorError("Not enough operands for GCD")
    
    b = calc.pop()
    a = calc.pop()
    
    try:
        # Convert floats to ints if they represent integers
        if isinstance(a, float) and a.is_integer():
            a = int(a)
        if isinstance(b, float) and b.is_integer():
            b = int(b)
        
        if not (isinstance(a, int) and isinstance(b, int)):
            raise ValueError("GCD requires two integers")
        
        result = math.gcd(a, b)
        calc.push(result)
    except Exception as e:
        calc.push(a)
        calc.push(b)
        raise CalculatorError(f"Error in GCD: {e}")

def _op_lcm(calc):
    if len(calc.stack) < 2:
        raise CalculatorError("Not enough operands for LCM")
    
    b = calc.pop()
    a = calc.pop()
    
    try:
        if isinstance(a, float) and a.is_integer():
            a = int(a)
        if isinstance(b, float) and b.is_integer():
            b = int(b)
        
        if not (isinstance(a, int) and isinstance(b, int)):
            raise ValueError("LCM requires two integers")
        
        if a == 0 or b == 0:
            result = 0
        else:
            result = abs(a * b) // math.gcd(a, b)
        
        calc.push(result)
    except Exception as e:
        calc.push(a)
        calc.push(b)
        raise CalculatorError(f"Error in LCM: {e}")
        
def _op_frac(calc):
    """Convert floating point to fraction with residual."""
    if not calc.stack:
        raise CalculatorError("FRAC needs a value")
    
    x = calc.pop()
    
    # Handle complex numbers - convert real and imaginary parts separately
    if isinstance(x, complex):
        calc.push(x)
        print("Converting real part:")
        calc.push(x.real)
        _op_frac(calc)
        print("Converting imaginary part:")
        calc.push(x.imag)
        _op_frac(calc)
        return
    
    # Handle vectors
    if isinstance(x, (list, tuple)):
        calc.push(x)
        raise CalculatorError("FRAC cannot process vectors")
    
    try:
        # Handle integers and zero
        if isinstance(x, int) or x == 0:
            print(f"{x} = {int(x)}/1")
            calc.push(x)
            return
        
        # Store sign and work with absolute value
        sign = 1 if x >= 0 else -1
        x_abs = abs(x)
        
        # Use continued fractions to find best rational approximation
        # with denominator up to max_denom
        max_denom = 1000000
        
        # Continued fractions algorithm
        n0, d0 = 0, 1
        n1, d1 = 1, 0
        
        remaining = x_abs
        
        for _ in range(50):  # Limit iterations
            if remaining == 0:
                break
                
            a = int(remaining)
            
            n2 = a * n1 + n0
            d2 = a * d1 + d0
            
            if d2 > max_denom:
                break
            
            n0, d0 = n1, d1
            n1, d1 = n2, d2
            
            remaining = remaining - a
            if remaining == 0:
                break
            remaining = 1.0 / remaining
        
        # Apply sign to numerator
        numerator = sign * n1
        denominator = d1
        
        # Calculate residual
        fraction_value = numerator / denominator
        residual = x - fraction_value
        
        # Format output
        if abs(residual) < 1e-15:  # Essentially exact
            print(f"{x} = {numerator}/{denominator}")
        else:
            # Format residual in scientific notation
            if residual == 0:
                residual_str = "0"
            else:
                residual_str = f"{residual:.6e}"
            print(f"{x} = {numerator}/{denominator} + {residual_str}")
        
        # Push the fraction value back (not the original)
        calc.push(fraction_value)
        
    except Exception as e:
        calc.push(x)
        raise CalculatorError(f"Error in FRAC: {e}")