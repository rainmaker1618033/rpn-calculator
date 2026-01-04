# Theory of Operation

## Overview

This document describes the internal architecture, design decisions, and operational theory of the RPN Calculator.

## Table of Contents

1. [Architecture](#architecture)
2. [Core Components](#core-components)
3. [Stack Management](#stack-management)
4. [Operation Registration](#operation-registration)
5. [Token Processing](#token-processing)
6. [Error Handling](#error-handling)
7. [Type System](#type-system)
8. [FFT Implementation](#fft-implementation)
9. [Logging System](#logging-system)
10. [Extension Points](#extension-points)

---

## Architecture

### Design Principles

The calculator follows these key design principles:

1. **Modularity**: Each operation category is in its own module
2. **Separation of Concerns**: Core logic, UI, and operations are separate
3. **Extensibility**: New operations can be added without modifying core
4. **Type Flexibility**: Supports scalars, complex numbers, vectors, and matrices
5. **Error Recovery**: Stack is preserved on errors when possible

### Module Organization

```
rpn_calculator/
├── Core Engine
│   ├── core.py              # Calculator state and orchestration
│   ├── errors.py            # Exception definitions
│   └── utils.py             # Shared utility functions
│
├── Operation Modules (each registers operations)
│   ├── arithmetic.py
│   ├── trigonometry.py
│   ├── complex_numbers.py
│   ├── vectors.py
│   ├── matrices.py
│   ├── matrix_decompositions.py
│   ├── fft_operations.py
│   ├── statistics.py
│   ├── integer_ops.py
│   └── calc_constants.py
│
├── User Interface
│   ├── cli.py               # Command-line interface
│   ├── help_text.py         # Documentation
│   └── formatting.py        # Output formatting
│
└── Supporting Systems
    └── logging.py           # Session logging
```

---

## Core Components

### CalculatorState

Holds configuration that affects operation behavior:

```python
@dataclass
class CalculatorState:
    degrees: bool = True      # Trig mode (DEG/RAD)
    digits: int = 6           # Display precision
    format: str = "FLOAT"     # Display format
```

### Calculator Class

Main orchestrator that:
- Maintains the stack
- Registers operations from all modules
- Processes tokens
- Manages history for UNDO
- Coordinates logging

Key data structures:
- `stack: List[Any]` - The RPN stack
- `operations: Dict[str, Callable]` - Operation registry
- `history: List[List[Any]]` - Stack snapshots for UNDO
- `logger: CalculatorLogger` - Session logger

---

## Stack Management

### The Stack

The stack is a Python list where:
- Index 0 is the bottom
- Index -1 is the top
- New items are appended to the end
- Operations pop from the end

### Stack Operations

**Basic Operations:**
- `push(value)` - Add to top
- `pop()` - Remove from top, returns value
- `peek()` - View top without removing
- `get_result()` - Safely peek (returns None if empty)

**Stack Manipulation:**
- `SWAP` - Exchange top two items
- `RD` (Roll Down) - Move top to bottom
- `RU` (Roll Up) - Move bottom to top

**History:**
- Snapshot taken before each operation
- `UNDO` restores previous snapshot
- History is a list of deep copies

---

## Operation Registration

### Registration Pattern

Each module defines:

```python
def register_operations(calc):
    """Register operations with calculator"""
    return {
        "OP_NAME": lambda: _op_function(calc),
        # ... more operations
    }
```

The lambda captures `calc` and calls the operation function.

### Registration Process

1. `Calculator.__init__` calls `_register_all_operations()`
2. For each module, calls `module.register_operations(self)`
3. Returned dict is merged into `self.operations`
4. Result: Single dictionary mapping command names to callables

### Operation Implementation Pattern

```python
def _op_example(calc):
    # 1. Validate stack has enough operands
    if len(calc.stack) < 2:
        raise CalculatorError("Need 2 operands")
    
    # 2. Pop operands (in reverse order!)
    b = calc.pop()
    a = calc.pop()
    
    try:
        # 3. Perform operation
        result = a + b
        
        # 4. Push result
        calc.push(result)
    except Exception as e:
        # 5. Restore stack on error
        calc.push(a)
        calc.push(b)
        raise CalculatorError(f"Error: {e}")
```

---

## Token Processing

### Tokenization

Input line is split on whitespace:
```
"3 4 +" → ["3", "4", "+"]
```

### Processing Loop

```python
def process_tokens(tokens):
    for each token:
        1. Check for special commands (HELP, DIGITS, etc.)
        2. Save history (if operation modifies stack)
        3. Try to execute as registered operation
        4. If not found, try to parse as number/vector/matrix
        5. Handle errors, preserving stack when possible
```

### Special Command Handling

Some commands are handled specially:
- `HELP` - May consume multiple tokens
- `DIGITS n` - Consumes next token
- `FORMAT FLOAT` - Consumes next token
- Mode commands (DEG, RAD) - Immediate execution

### Number Parsing

Parse priority:
1. **Vector/Matrix** - If starts with `[` and ends with `]`
2. **Integer** - If no `.`, `e`, or `j`
3. **Float/Complex** - Parse as complex, return real if imag=0

---

## Error Handling

### Error Philosophy

1. **Preserve Stack**: On error, restore stack to pre-operation state
2. **Clear Messages**: Tell user what went wrong
3. **No Crashes**: Catch and report errors gracefully

### CalculatorError

```python
@dataclass
class CalculatorError(Exception):
    message: str
    restore_stack: bool = True
```

- `message` - User-friendly error description
- `restore_stack` - Whether to restore stack (usually True)

### Error Propagation

1. Operation detects error
2. Restores operands to stack
3. Raises `CalculatorError`
4. Caught in `process_tokens`
5. Message printed, execution continues

---

## Type System

### Supported Types

1. **Scalar Numbers**
   - `int` - Integers
   - `float` - Real numbers
   - `complex` - Complex numbers (a+bj)

2. **Vectors**
   - Python `list` or `tuple` of numbers
   - Must be 1D (no nested lists)
   - Example: `[1, 2, 3]`

3. **Matrices**
   - Python `list` of lists
   - All rows must have same length
   - Example: `[[1,2], [3,4]]`

### Type Detection

```python
def is_vector(x):
    return isinstance(x, (list, tuple)) and x and \
           not isinstance(x[0], (list, tuple))

def is_matrix(x):
    return isinstance(x, (list, tuple)) and x and \
           isinstance(x[0], (list, tuple)) and \
           all(len(row) == len(x[0]) for row in x)
```

### Type Coercion

**Vector Operations:**
- Scalar + Vector → broadcast scalar to each element
- Vector + Vector → element-wise operation
- Must have matching lengths

**Matrix Operations:**
- Some ops require NumPy conversion
- Results converted back to Python lists
- Complex → real when imaginary part negligible

---

## FFT Implementation

### Zero-Padding Strategy

FFT requires power-of-2 length for efficiency:

```python
def _next_power_of_2(n):
    if n & (n-1) == 0:  # Already power of 2
        return n
    power = 1
    while power < n:
        power *= 2
    return power
```

Padding process:
1. Check current length
2. Find next power of 2
3. Append zeros to reach that length
4. Inform user if padding occurred

### FFT Operations

**FFT (Forward Transform):**
- Input: Real or complex vector
- Output: Complex frequency domain vector
- DC component (index 0) = sum of input

**IFFT (Inverse Transform):**
- Input: Complex frequency domain
- Output: Time/spatial domain
- Returns real values if imaginary parts negligible

**FFT_MAG (Magnitude Spectrum):**
- Computes |FFT(x)| for each bin
- Useful for frequency analysis
- Always returns real values

**FFT_PHASE (Phase Spectrum):**
- Computes angle(FFT(x)) for each bin
- Respects calculator's angle mode (DEG/RAD)
- Returns wrapped phase [-π,π] or [-180°,180°]

### Conjugate Symmetry

For real inputs, FFT exhibits conjugate symmetry:
- FFT[k] = conj(FFT[N-k])
- Only first N/2+1 bins needed for real signals
- Used in validation tests

---

## Logging System

### Log File Structure

```
logs/rpn_log_YYYYMMDD_HHMMSS.txt
```

### Log Entry Format

```
[HH:MM:SS] Input: <command>
  Stack before: [items]
  Stack after: [items]
  Result: <top item>
```

### Logging Flow

1. User enters command
2. Stack snapshot taken (deep copy)
3. Command executed
4. Logger records: input, before, after, errors
5. Entry written to file

### Log File Lifecycle

- **Created**: When Calculator initialized
- **Written**: After each command
- **Closed**: When calculator exits (finally block)
- **Contains**: Session start, all operations, session end

---

## Extension Points

### Adding New Operations

**1. Create Module**

```python
# rpn_calculator/my_operations.py
from .errors import CalculatorError

def register_operations(calc):
    return {
        "MYOP": lambda: _my_op(calc),
    }

def _my_op(calc):
    # Implementation
    pass
```

**2. Register in Core**

```python
# core.py
from . import my_operations

modules = [
    # ... existing modules
    my_operations,
]
```

**3. Add Help**

```python
# help_text.py
HELP_SECTIONS["MyOps"] = '''
My Operations:
  MYOP    Description of operation
'''
```

**4. Add Tests**

```python
# tests/test_my_operations.py
class TestMyOperations(unittest.TestCase):
    def test_myop(self):
        calc = Calculator(enable_logging=False)
        calc.push(5)
        calc.operations["MYOP"]()
        result = calc.get_result()
        self.assertEqual(result, expected)
```

### Custom Number Types

To add new numeric types:

1. Extend type detection in `utils.py`
2. Add handling in operation modules
3. Update `_format_value` in `formatting.py`
4. Add parsing in `_parse_and_push`

### Custom Display Formats

To add new display formats:

1. Add format option to `CalculatorState`
2. Update `ValueFormatter.format_value()`
3. Add FORMAT command handler in `cli.py`

---

## Performance Considerations

### Memory

- Deep copies for history can be expensive
- History limited to recent operations (no limit currently)
- Consider adding history size limit for very long sessions

### Computation

- NumPy used for matrix operations (efficient)
- FFT is O(N log N) - efficient for large vectors
- Vector operations use list comprehensions (pure Python)

### Optimization Opportunities

1. **History Management**: Limit history depth
2. **Matrix Storage**: Keep as NumPy arrays internally
3. **Lazy Evaluation**: Defer computation until needed
4. **Caching**: Cache FFT results for repeated operations

---

## Security Considerations

### Input Validation

- All user input is tokenized and parsed
- No `eval()` of arbitrary code
- `ast.literal_eval()` used for safe list parsing

### File Operations

- Logs written to dedicated directory
- Timestamped filenames prevent collisions
- No user-controlled file paths

### Error Messages

- Don't expose internal paths or system info
- User-friendly messages only

---

## Future Enhancements

### Potential Features

1. **Scripting**: Load and execute command files
2. **Variables**: Store and recall values by name
3. **Functions**: User-defined operations
4. **Plotting**: Integrate matplotlib for visualization
5. **2D FFT**: For image processing
6. **Symbolic Math**: Integration with SymPy
7. **Units**: Physical unit support
8. **Database**: Persist session across restarts

### Plugin System

Future architecture could support plugins:

```python
class Plugin:
    def register_operations(self, calc):
        return {...}
    
    def register_help(self):
        return {...}
```

Plugins loaded from directory at startup.

---

## References

### Mathematical Foundations

- **RPN**: Reverse Polish Notation (postfix notation)
- **FFT**: Cooley-Tukey algorithm (O(N log N))
- **Matrix Decompositions**: Standard linear algebra techniques

### Implementation References

- NumPy documentation: https://numpy.org/doc/
- SciPy documentation: https://docs.scipy.org/
- Python typing: https://docs.python.org/3/library/typing.html

### HP Calculator Inspiration

This calculator is inspired by HP RPN calculators, particularly:
- HP-15C (scientific functions)
- HP-48 series (complex, matrices, programming)

---

## Appendix: Code Conventions

### Naming

- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`
- Private methods: `_leading_underscore`

### Module Structure

```python
# 1. Docstring
# 2. Imports
# 3. register_operations() function
# 4. Operation implementations
```

### Error Handling

```python
# Always restore stack on error
try:
    result = operation(a, b)
    calc.push(result)
except Exception as e:
    calc.push(a)
    calc.push(b)
    raise CalculatorError(f"Error: {e}")
```

---

*Last Updated: January 2026*
