# ============================================================================
# FILE: rpn_calculator/core.py
# ============================================================================
"""Core Calculator class and state management"""

import copy
from typing import List, Any, Dict, Callable
from dataclasses import dataclass, field

from .errors import CalculatorError
from .utils import is_vector, is_matrix

# Import operation modules
from . import stack_operations
from . import arithmetic
from . import trigonometry
from . import logarithmic
from . import complex_numbers
from . import calc_constants
from . import vectors
from . import matrices
from . import matrix_decompositions
from . import statistics
from . import integer_ops
from . import calc_constants
from . import fft_operations
from .logging import CalculatorLogger
from .formatting import ValueFormatter

@dataclass
class CalculatorState:
    """Holds calculator configuration state."""
    degrees: bool = True
    digits: int = 6
    format: str = "FLOAT"

class Calculator:
    """RPN Calculator with complex numbers, vectors, and matrices."""
    
    def __init__(self,enable_logging=True):
        self.stack: List[Any] = []
        self.state = CalculatorState()
        self.history: List[List[Any]] = []
        self.operations: Dict[str, Callable] = {}
        self.formatter = ValueFormatter(self.state)
        #self.logger = CalculatorLogger(enabled=True)

        # Only enable logging if requested (disabled for tests and library use)
        if enable_logging:
            from .logging import CalculatorLogger
            self.logger = CalculatorLogger(enabled=True)
        else:
            # Create a dummy logger that does nothing
            class DummyLogger:
                def __init__(self):
                    self.enabled = False
                def log_operation(self, *args, **kwargs):
                    pass
                def log_message(self, *args, **kwargs):
                    pass
                def close(self):
                    pass
            self.logger = DummyLogger()
		
        # Utility methods available to operation modules
        self.is_vector = is_vector
        self.is_matrix = is_matrix
        
        # Build operation registry from all modules
        self._register_all_operations()
    
    def _register_all_operations(self):
        """Register operations from all modules"""
        modules = [
            stack_operations,
            arithmetic,
            trigonometry,
            logarithmic,
            complex_numbers,
            vectors,
            matrices,
            matrix_decompositions,
            statistics,
            integer_ops,
            calc_constants,
            fft_operations,
        ]
        
        for module in modules:
            ops = module.register_operations(self)
            self.operations.update(ops)
    
    def _save_history(self):
        """Save current stack state to history."""
        self.history.append(copy.deepcopy(self.stack))
    
    def _should_save_history(self, token: str) -> bool:
        """Determine if operation should save history."""
        no_history_ops = {"H", "HELP", "SHOWMODE", "UNDO", "DIGITS", "FORMAT"}
        return token not in no_history_ops
    
    # Helper methods used by operation modules
    def push(self, value: Any):
        """Push value onto stack"""
        self.stack.append(value)
    
    def pop(self) -> Any:
        """Pop and return top of stack"""
        if not self.stack:
            raise CalculatorError("Stack is empty", restore_stack=False)
        return self.stack.pop()
    
    def peek(self) -> Any:
        """Return top of stack without popping"""
        if not self.stack:
            raise CalculatorError("Stack is empty", restore_stack=False)
        return self.stack[-1]
    
    def get_result(self) -> Any:
        """Get top of stack without popping (returns None if empty)"""
        return self.stack[-1] if self.stack else None
    
    def evaluate(self, expression: str) -> Any:
        """Evaluate RPN expression and return result"""
        tokens = expression.split()
        self.process_tokens(tokens)
        return self.get_result()
    
    def evaluate_and_clear(self, expression: str) -> Any:
        """Evaluate and return result, clearing stack after"""
        result = self.evaluate(expression)
        self.stack.clear()
        return result
    
    # Token processing
    def process_tokens(self, tokens: List[str]):
        '''Process a list of RPN tokens.'''
        from .cli import handle_special_commands
        
        skip_next = False
        i = 0
        while i < len(tokens):
            if skip_next:
                skip_next = False
                i += 1
                continue
            
            token = tokens[i].strip().upper()
            if not token:
                i += 1
                continue
            
            # Save stack state before operation (for logging)
            stack_before = copy.deepcopy(self.stack)
            error_msg = None
            
            try:
                # Handle special commands (HELP, DIGITS, FORMAT)
                result = handle_special_commands(self, token, tokens, i)
                if result == "skip_all":
                    break
                elif result == "skip_next":
                    skip_next = True
                    i += 1
                    continue
                elif result == "handled":
                    i += 1
                    continue
                
                # Save history before operations
                if self._should_save_history(token):
                    self._save_history()
                
                # Try to execute operation
                if token in self.operations:
                    self.operations[token]()
                else:
                    # Try to parse as number or vector/matrix
                    self._parse_and_push(token)
                    
            except CalculatorError as e:
                error_msg = e.message
                print(f"Error: {e.message}")
            
            # Log the operation
            # self.logger.log_operation(token, stack_before, self.stack, error_msg)
            
            i += 1

    
    def _parse_and_push(self, token: str):
        """Parse and push a number, vector, or matrix onto the stack."""
        import ast
        try:
            if token.startswith("[") and token.endswith("]"):
                parsed = ast.literal_eval(token)
                if not isinstance(parsed, (list, tuple)):
                    raise ValueError("Invalid vector/matrix format")
                self.stack.append(parsed)
            else:
                # Try to parse as integer first
                try:
                    # Check if it's an integer (no decimal point, no 'e' notation)
                    if '.' not in token and 'e' not in token.lower() and 'j' not in token.lower():
                        val = int(token)
                        self.stack.append(val)
                        return
                except ValueError:
                    pass
                
                # Otherwise parse as complex/float
                val = complex(token.replace('I', 'j'))
                if val.imag == 0:
                    val = val.real
                self.stack.append(val)
        except Exception:
            raise CalculatorError(f"Invalid token: {token}", restore_stack=False)
        
    def print_stack(self):
        """Print the current stack."""
        mode = "DEG" if self.state.degrees else "RAD"
        if not self.stack:
            print(f"[Stack empty]   (Mode: {mode})")
        else:
            print(f"\nStack (Mode: {mode}, {self.state.digits} digits, {self.state.format}):")
            for i, val in enumerate(reversed(self.stack), 1):
                print(f"{i:>2}: {self.formatter.format_value(val)}")
            print()
    
    def print_stack_inline(self):
        """Print stack inline."""
        if not self.stack:
            print("[Stack empty]")
            return
        formatted = [self.formatter.format_value(v) for v in reversed(self.stack)]
        print("Stack (inline): Top → " + "  ".join(formatted) + " ← Bottom")
