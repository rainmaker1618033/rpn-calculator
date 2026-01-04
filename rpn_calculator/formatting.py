# ============================================================================
# FILE: rpn_calculator/formatting.py
# ============================================================================
"""Output formatting for different value types"""

class ValueFormatter:
    """Handles formatting of values for display"""
    
    def __init__(self, state):
        self.state = state
    
    def format_value(self, val) -> str:
        """Format a value for display."""
        digits = self.state.digits
        fmt_mode = self.state.format
        fmt = f".{digits}{'e' if fmt_mode == 'SCIENTIFIC' else 'f'}"
        
        def format_number(x):
            if isinstance(x, int):
                return str(x)
            try:
                formatted = format(x, fmt)
                return formatted.rstrip('0').rstrip('.') if fmt_mode == 'FLOAT' else formatted
            except Exception:
                return str(x)
        
        if isinstance(val, complex):
            real = format_number(val.real)
            imag = format_number(abs(val.imag))
            sign = "+" if val.imag >= 0 else "-"
            return f"{real}{sign}{imag}j"
        elif isinstance(val, (float, int)):
            return format_number(val)
        elif isinstance(val, (list, tuple)):
            return "[" + ", ".join(self.format_value(v) for v in val) + "]"
        else:
            return str(val)