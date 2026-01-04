# ============================================================================
# FILE: rpn_calculator/__init__.py
# ============================================================================
"""
RPN Calculator Package
A modular scientific RPN calculator with support for complex numbers,
vectors, matrices, and advanced operations.
"""

from .core import Calculator, CalculatorState
from .errors import CalculatorError
from .cli import main

__version__ = "1.0.0"
__all__ = ["Calculator", "CalculatorState", "CalculatorError", "main"]