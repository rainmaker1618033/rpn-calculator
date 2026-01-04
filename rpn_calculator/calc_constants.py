# ============================================================================
# FILE: rpn_calculator/calc_constants.py
# ============================================================================
"""Mathematical constants"""

import math

def register_operations(calc):
    """Register constant operations"""
    return {
        "E": lambda: calc.push(math.e),
        "PI": lambda: calc.push(math.pi),
        "I": lambda: calc.push(1j),
    }