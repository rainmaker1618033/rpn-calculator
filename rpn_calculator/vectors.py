"""Vector operations"""

import math
from .errors import CalculatorError

def register_operations(calc):
    """Register vector operations"""
    return {
        "DOT": lambda: _op_dot(calc),
        "VMAG": lambda: _op_vmag(calc),
        "VCROSS": lambda: _op_vcross(calc),
        "VNORM": lambda: _op_vnorm(calc),
    }

def _op_dot(calc):
    if len(calc.stack) < 2:
        raise CalculatorError("DOT needs 2 vectors")
    b = calc.pop()
    a = calc.pop()
    try:
        if not (hasattr(a, '__iter__') and hasattr(b, '__iter__')):
            raise ValueError("Both operands must be vectors")
        if len(a) != len(b):
            raise ValueError("Vectors must be of the same length")
        result = sum(x * y for x, y in zip(a, b))
        calc.push(result)
    except Exception as e:
        calc.push(a)
        calc.push(b)
        raise CalculatorError(f"Error in DOT: {e}")

def _op_vmag(calc):
    if not calc.stack:
        raise CalculatorError("VMAG needs a vector")
    v = calc.pop()
    if not isinstance(v, (list, tuple)):
        calc.push(v)
        raise CalculatorError("VMAG expects a vector")
    try:
        mag = math.sqrt(sum((abs(x) ** 2 for x in v)))
        calc.push(mag)
    except Exception as e:
        calc.push(v)
        raise CalculatorError(f"Error in VMAG: {e}")

def _op_vcross(calc):
    if len(calc.stack) < 2:
        raise CalculatorError("VCROSS needs 2 vectors")
    b = calc.pop()
    a = calc.pop()
    if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
        calc.push(a)
        calc.push(b)
        raise CalculatorError("VCROSS expects 2 vectors")
    if len(a) != 3 or len(b) != 3:
        calc.push(a)
        calc.push(b)
        raise CalculatorError("VCROSS requires 3D vectors")
    try:
        cx = a[1] * b[2] - a[2] * b[1]
        cy = a[2] * b[0] - a[0] * b[2]
        cz = a[0] * b[1] - a[1] * b[0]
        calc.push([cx, cy, cz])
    except Exception as e:
        calc.push(a)
        calc.push(b)
        raise CalculatorError(f"Error in VCROSS: {e}")

def _op_vnorm(calc):
    if not calc.stack:
        raise CalculatorError("VNORM needs a vector")
    v = calc.pop()
    if not isinstance(v, (list, tuple)):
        calc.push(v)
        raise CalculatorError("VNORM expects a vector")
    try:
        mag = math.sqrt(sum((x ** 2 for x in v)))
        if mag == 0:
            calc.push(v)
            raise CalculatorError("Cannot normalize zero vector")
        norm_v = [x / mag for x in v]
        calc.push(norm_v)
    except CalculatorError:
        raise
    except Exception as e:
        calc.push(v)
        raise CalculatorError(f"Error in VNORM: {e}")
