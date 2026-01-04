# ============================================================================
# FILE: rpn_calculator/arithmetic.py
# ============================================================================
"""Basic arithmetic operations"""

from .errors import CalculatorError
from .utils import is_vector

def register_operations(calc):
    """Register arithmetic operations"""
    return {
        "+": lambda: _binary_op(calc, lambda a, b: a + b, "+"),
        "-": lambda: _binary_op(calc, lambda a, b: a - b, "-"),
        "*": lambda: _binary_op(calc, lambda a, b: a * b, "*"),
        "/": lambda: _binary_op(calc, lambda a, b: _safe_divide(a, b), "/"),
        "^": lambda: _binary_op(calc, lambda a, b: a ** b, "^"),
        "MOD": lambda: _binary_op(calc, lambda a, b: a % b, "MOD"),
        "||": lambda: _op_parallel(calc),
    }

def _safe_divide(a, b):
    if b == 0:
        raise CalculatorError("Divide by zero", restore_stack=True)
    return a / b

def _binary_op(calc, op, name: str):
    """Execute a binary operation with vector support."""
    if len(calc.stack) < 2:
        raise CalculatorError(f"Not enough operands for '{name}'", restore_stack=False)
    
    b = calc.pop()
    a = calc.pop()
    
    try:
        if is_vector(a) or is_vector(b):
            a_vec = list(a) if is_vector(a) else [a] * (len(b) if is_vector(b) else 1)
            b_vec = list(b) if is_vector(b) else [b] * len(a_vec)
            
            if len(a_vec) != len(b_vec):
                raise ValueError("Vector length mismatch")
            
            result = [op(x, y) for x, y in zip(a_vec, b_vec)]
            calc.push(result)
        else:
            result = op(a, b)
            calc.push(result)
    except CalculatorError as e:
        calc.push(a)
        calc.push(b)
        raise
    except Exception as e:
        calc.push(a)
        calc.push(b)
        raise CalculatorError(f"Error in operation '{name}': {e}", restore_stack=False)

def _op_parallel(calc):
    """Product over Sum: (x*y)/(x+y)"""
    if len(calc.stack) < 2:
        raise CalculatorError("'||' needs 2 values")
    
    y = calc.pop()
    x = calc.pop()
    
    if is_vector(x) or is_vector(y):
        calc.push(x)
        calc.push(y)
        raise CalculatorError("'||' cannot process vectors")
    
    try:
        if x + y == 0:
            raise ValueError("Sum is zero")
        z = (x * y) / (x + y)
        calc.push(z)
    except Exception as e:
        calc.push(x)
        calc.push(y)
        raise CalculatorError(f"Error in '||': {e}")