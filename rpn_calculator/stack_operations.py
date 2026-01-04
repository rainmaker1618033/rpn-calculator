# ============================================================================
# FILE: rpn_calculator/stack_operations.py
# ============================================================================
"""Stack manipulation operations"""

from .errors import CalculatorError

def register_operations(calc):
    """Register stack operations"""
    return {
        "C": lambda: _op_clear(calc),
        "DEL": lambda: _op_delete(calc),
        "UNDO": lambda: _op_undo(calc),
        "RD": lambda: _op_roll_down(calc),
        "RU": lambda: _op_roll_up(calc),
        "R↓": lambda: _op_roll_down(calc),
        "R↑": lambda: _op_roll_up(calc),
        "X<>Y": lambda: _op_swap(calc),
        "SWAP": lambda: _op_swap(calc),
    }

def _op_clear(calc):
    calc.stack.clear()
    print("Stack cleared.")

def _op_delete(calc):
    if not calc.stack:
        raise CalculatorError("Stack is empty — nothing to delete.", restore_stack=False)
    removed = calc.stack.pop()
    print(f"Deleted top of stack: {removed}")

def _op_undo(calc):
    if not calc.history:
        raise CalculatorError("Nothing to undo.", restore_stack=False)
    last_state = calc.history.pop()
    calc.stack.clear()
    calc.stack.extend(last_state)
    print("Undo successful. Stack restored to previous state.")

def _op_roll_down(calc):
    if len(calc.stack) < 2:
        raise CalculatorError("Need at least 2 stack levels to roll down.")
    calc.stack.insert(0, calc.stack.pop())

def _op_roll_up(calc):
    if len(calc.stack) < 2:
        raise CalculatorError("Need at least 2 stack levels to roll up.")
    calc.stack.append(calc.stack.pop(0))

def _op_swap(calc):
    if len(calc.stack) < 2:
        raise CalculatorError("Need at least 2 stack levels to swap.")
    calc.stack[-1], calc.stack[-2] = calc.stack[-2], calc.stack[-1]