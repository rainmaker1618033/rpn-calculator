# ============================================================================
# FILE: rpn_calculator/logging.py
# ============================================================================
"""Logging functionality for calculator operations"""

import os
from datetime import datetime
from typing import Any, List

class CalculatorLogger:
    """Handles logging of calculator operations to file"""
    
    def __init__(self, log_directory="logs", enabled=True):
        """
        Initialize logger
        
        Args:
            log_directory: Directory to store log files
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.log_directory = log_directory
        self.log_file = None
        self.session_start = None
        
        if self.enabled:
            self._create_log_file()
    
    def _create_log_file(self):

        #print(f"DEBUG: Current working directory: {os.getcwd()}")
        #print(f"DEBUG: Log directory will be: {self.log_directory}")
        full_path = os.path.join(os.getcwd(), self.log_directory) 
        #print(f"DEBUG: Full log path: {full_path}")

        """Create a new log file with timestamp"""
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
        
        # Generate timestamp for filename
        self.session_start = datetime.now()
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        filename = f"rpn_log_{timestamp}.txt"
        self.log_file = os.path.join(self.log_directory, filename)
        
        # Write header
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RPN CALCULATOR SESSION LOG\n")
            f.write(f"Session started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
        
        #print(f"Logging to: {self.log_file}")
        print(f"Logging to: {full_path}")
    
    def log_operation(self, input_line: str, stack_before: List[Any], 
                     stack_after: List[Any], error: str = None):
        """
        Log an operation
        
        Args:
            input_line: The input command/expression
            stack_before: Stack state before operation
            stack_after: Stack state after operation
            error: Error message if operation failed
        """
        if not self.enabled or not self.log_file:
            return
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Log timestamp and input
                f.write(f"[{timestamp}] Input: {input_line}\n")
                
                # Log stack before
                if stack_before:
                    f.write(f"  Stack before: {self._format_stack(stack_before)}\n")
                else:
                    f.write(f"  Stack before: [empty]\n")
                
                # Log error or result
                if error:
                    f.write(f"  ERROR: {error}\n")
                else:
                    if stack_after:
                        f.write(f"  Stack after:  {self._format_stack(stack_after)}\n")
                        # Highlight what changed/was added
                        if len(stack_after) > len(stack_before):
                            new_items = stack_after[len(stack_before):]
                            f.write(f"  Result: {self._format_value(new_items[-1])}\n")
                        elif stack_after:
                            f.write(f"  Result: {self._format_value(stack_after[-1])}\n")
                    else:
                        f.write(f"  Stack after:  [empty]\n")
                
                f.write("\n")  # Blank line between operations
        except Exception as e:
            print(f"Warning: Failed to write to log file: {e}")
    
    def log_message(self, message: str):
        """Log a general message"""
        if not self.enabled or not self.log_file:
            return
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%H:%M:%S")
                f.write(f"[{timestamp}] {message}\n\n")
        except Exception as e:
            print(f"Warning: Failed to write to log file: {e}")
    
    def _format_stack(self, stack: List[Any]) -> str:
        """Format stack for display"""
        if not stack:
            return "[empty]"
        return "[" + ", ".join(self._format_value(v) for v in stack) + "]"
    
    def _format_value(self, val: Any) -> str:
        """Format a single value for display"""
        if isinstance(val, complex):
            if val.imag >= 0:
                return f"{val.real}+{val.imag}j"
            else:
                return f"{val.real}{val.imag}j"
        elif isinstance(val, float):
            # Limit decimal places for readability
            return f"{val:.6g}"
        elif isinstance(val, (list, tuple)):
            # FIX THIS -- need cleaner display for long vectors
            # Truncate long vectors/matrices

            #if len(val) > 5:
            #    return f"[{len(val)} elements]"
			
            return str(val)
        else:
            return str(val)
    
    def close(self):
        """Close the log file and write footer"""
        if not self.enabled or not self.log_file:
            return
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                session_end = datetime.now()
                f.write(f"Session ended: {session_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                if self.session_start:
                    duration = session_end - self.session_start
                    f.write(f"Session duration: {duration}\n")
                
                f.write("="*80 + "\n")
            
            print(f"Log saved to: {self.log_file}")
        except Exception as e:
            print(f"Warning: Failed to close log file: {e}")


