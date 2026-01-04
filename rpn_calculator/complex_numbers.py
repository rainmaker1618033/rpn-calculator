"""Complex number operations"""

import math
import cmath
from .errors import CalculatorError

def register_operations(calc):
    """Register complex number operations"""
    return {
        "CMPLX": lambda: _op_cmplx(calc),
        "RECT": lambda: _op_rect(calc),
        "POLAR": lambda: _op_polar(calc),
        "RE": lambda: _op_re(calc),
        "IM": lambda: _op_im(calc),
        "ABS": lambda: _op_abs(calc),
        "ARG": lambda: _op_arg(calc),
        "CONJ": lambda: _op_conj(calc),
    }

def _op_cmplx(calc):
    if len(calc.stack) < 2:
        raise CalculatorError("CMPLX needs 2 values (real, imag)")
    
    imag = calc.pop()
    real = calc.pop()
    z = complex(real, imag)
    calc.push(z)

def _op_rect(calc):
    if len(calc.stack) < 2:
        raise CalculatorError("RECT needs 2 values (r, theta)")
    
    theta = calc.pop()
    r = calc.pop()
    
    if calc.state.degrees:
        theta = math.radians(theta)
    
    z = cmath.rect(r, theta)
    calc.push(z)

def _op_polar(calc):
    if not calc.stack:
        raise CalculatorError("POLAR needs a complex number")
    
    z = calc.pop()
    r, theta = cmath.polar(z)
    
    if calc.state.degrees:
        theta = math.degrees(theta)
    
    calc.push(r)
    calc.push(theta)

def _op_re(calc):
    if not calc.stack:
        raise CalculatorError("RE needs a value")
    z = calc.pop()
    calc.push(complex(z).real)

def _op_im(calc):
    if not calc.stack:
        raise CalculatorError("IM needs a value")
    z = calc.pop()
    calc.push(complex(z).imag)

def _op_abs(calc):
    if not calc.stack:
        raise CalculatorError("ABS needs a value")
    z = calc.pop()
    calc.push(abs(z))

def _op_arg(calc):
    if not calc.stack:
        raise CalculatorError("ARG needs a value")
    z = calc.pop()
    angle = cmath.phase(z)
    if calc.state.degrees:
        angle = math.degrees(angle)
    calc.push(angle)

def _op_conj(calc):
    if not calc.stack:
        raise CalculatorError("CONJ needs a value")
    z = calc.pop()
    calc.push(complex(z).conjugate())