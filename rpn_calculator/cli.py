# ============================================================================
# FILE: rpn_calculator/cli.py
# ============================================================================
"""Command-line interface and help system"""

def handle_special_commands(calc, token, tokens, index):
    """
    Handle special commands that don't fit the normal operation pattern.
    Returns: "handled", "skip_next", or None
    """
    # Handle HELP
    if token in {"H", "HELP"}:
        # Consume the rest of the line as the help argument
        if index + 1 < len(tokens):
            # Join all remaining tokens as the help topic
            remaining = " ".join(tokens[index + 1:])
            show_instructions(remaining)
            # Mark all remaining tokens as consumed
            for j in range(index + 1, len(tokens)):
                tokens[j] = ""  # Clear the tokens so they won't be processed
        else:
            show_instructions(None)
        return "handled"
    
    # Handle DIGITS
    if token == "DIGITS":
        if index + 1 < len(tokens) and tokens[index + 1].isdigit():
            calc.state.digits = int(tokens[index + 1])
            print(f"Display precision set to {calc.state.digits} digits.")
            return "skip_next"
        else:
            print("Usage: DIGITS n   (e.g., DIGITS 4)")
        return "handled"
    
    # Handle FORMAT
    if token == "FORMAT":
        if index + 1 < len(tokens):
            fmt = tokens[index + 1].upper()
            if fmt in {"FLOAT", "SCIENTIFIC"}:
                calc.state.format = fmt
                print(f"Number format set to: {fmt}")
                return "skip_next"
            else:
                print("Usage: FORMAT FLOAT | FORMAT SCIENTIFIC")
        else:
            print("Usage: FORMAT FLOAT | FORMAT SCIENTIFIC")
        return "handled"
    
    # Handle mode operations
    if token == "DEG":
        calc.state.degrees = True
        print("Mode set to: Degrees")
        return "handled"
    
    if token == "RAD":
        calc.state.degrees = False
        print("Mode set to: Radians")
        return "handled"
    
    if token == "SHOWMODE":
        print(f"Current mode: {'Degrees' if calc.state.degrees else 'Radians'}")
        print(f"Display: {calc.state.digits} digits, {calc.state.format} format")
        return "handled"
    
    if token == "LOG":
        if index + 1 < len(tokens):
            action = tokens[index + 1].upper()
            if action == "ON":
                calc.logger.enabled = True
                calc.logger.log_message("Logging enabled")
                print("Logging enabled")
                return "skip_next"
            elif action == "OFF":
                calc.logger.enabled = False
                print("Logging disabled")
                return "skip_next"
        print(f"Logging is currently: {'ON' if calc.logger.enabled else 'OFF'}")
        print("Usage: LOG ON | LOG OFF")
        return "handled"
    
    return None

def show_instructions(choice=None):
    """Display calculator instructions"""
    # Import help sections from modules
    from . import help_text
    help_text.show_instructions(choice)

def main():
    '''Main CLI loop'''
    from .core import Calculator
    import copy
    
    show_instructions(None)
    calc = Calculator()
    
    try:
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\\nExiting.")
                break
            
            if line == "@":
                print("Goodbye.")
                break
            
            if line == "":
                if calc.stack:
                    calc.stack.append(calc.stack[-1])
                calc.print_stack()
                continue
            
            # Save stack state before operation (for logging)
            stack_before = copy.deepcopy(calc.stack)
            error_msg = None
            
            try:
                tokens = line.split()
                calc.process_tokens(tokens)
            except Exception as e:
                error_msg = str(e)

            # Log the operation
            calc.logger.log_operation(line, stack_before, calc.stack, error_msg)

            # DEBUG Log the operation
            #print(f"DEBUG: About to log operation: '{line}'")
            #print(f"DEBUG: Stack before: {stack_before}")
            #print(f"DEBUG: Stack after: {calc.stack}")
            #calc.logger.log_operation(line, stack_before, calc.stack, error_msg)
            #print(f"DEBUG: Logged successfully")

            calc.print_stack()
            calc.print_stack_inline()
    
    finally:
        # Close logger when exiting
        calc.logger.close()


# ============================================================================
# SUMMARY: File Organization
# ============================================================================
"""
Complete file structure:

rpn_calculator/
├── __init__.py                 # Package exports
├── core.py                     # Calculator class, state, main logic
├── errors.py                   # CalculatorError exception
├── utils.py                    # is_vector, is_matrix helpers
├── formatting.py               # ValueFormatter class
├── cli.py                      # main(), command handling, help
├── help_text.py                # Help documentation
│
├── stack_operations.py         # C, DEL, UNDO, SWAP, RD, RU
├── arithmetic.py               # +, -, *, /, ^, MOD
├── trigonometry.py             # SIN, COS, TAN, ASIN, ACOS, ATAN
├── logarithmic.py              # LOG, LN, EXP, SQRT, 1/X
├── complex_numbers.py          # CMPLX, RECT, POLAR, RE, IM, ABS, ARG, CONJ
├── vectors.py                  # DOT, VMAG, VCROSS, VNORM, ||
├── matrices.py                 # MATRIX, DET, TRACE, MINV, M+, M-, M*, etc.
├── matrix_decompositions.py   # LU, QR, SVD, CHOLESKY, SCHUR, etc.
├── statistics.py               # COMB, PERM, STDV
└── integer_ops.py              # GCD, LCM

Usage:
    from rpn_calculator import Calculator
    
    calc = Calculator()
    result = calc.evaluate("3 4 +")
    print(result)  # 7

Or run CLI:
    python -m rpn_calculator
"""