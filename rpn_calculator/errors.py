# ============================================================================
# FILE: rpn_calculator/errors.py
# ============================================================================
"""Custom exceptions for the calculator"""

from dataclasses import dataclass

@dataclass
class CalculatorError(Exception):
    """Custom exception for calculator errors."""
    message: str
    restore_stack: bool = True
