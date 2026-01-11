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
9. [Signal Processing](#signal-processing)
10. [Logging System](#logging-system)
11. [Extension Points](#extension-points)

---

## Architecture

### Design Principles

The calculator follows these key design principles:

1. **Modularity**: Each operation category is in its own module
2. **Separation of Concerns**: Core logic, UI, and operations are separate
3. **Extensibility**: New operations can be added without modifying core
4. **Type Flexibility**: Supports scalars, complex numbers, vectors, and matrices
5. **Error Recovery**: Stack is preserved on errors when possible
6. **Performance**: NumPy/SciPy for computationally intensive operations

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
│   ├── logarithmic.py
│   ├── complex_numbers.py
│   ├── vectors.py
│   ├── matrices.py
│   ├── matrix_decompositions.py
│   ├── fft_operations.py
│   ├── signal_processing.py      # NEW: Convolution operations
│   ├── statistics.py
│   ├── integer_ops.py
│   ├── stack_operations.py
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
    format: str = "FLOAT"     # Display format (FLOAT/SCIENTIFIC)
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
- `state: CalculatorState` - Configuration state
- `logger: CalculatorLogger` - Session logger
- `formatter: ValueFormatter` - Output formatting

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
- `SWAP` (X<>Y) - Exchange top two items
- `RD` (Roll Down) - Move top to bottom, shift others up
- `RU` (Roll Up) - Move bottom to top, shift others down
- `DEL` - Delete top item
- `C` - Clear entire stack
- `UNDO` - Restore previous stack state

**History:**
- Snapshot taken before each modifying operation
- `UNDO` restores previous snapshot
- History is a list of deep copies
- Not all operations save history (e.g., HELP, SHOWMODE)

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

**Note**: Always pop in reverse order! For `a + b`, pop `b` first, then `a`.

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
        6. Log operation (input, before, after, errors)
```

### Special Command Handling

Some commands are handled specially:
- `HELP [topic]` - May consume multiple tokens
- `DIGITS n` - Consumes next token as integer
- `FORMAT FLOAT|SCIENTIFIC` - Consumes next token
- `LOG ON|OFF` - Consumes next token
- Mode commands (DEG, RAD, SHOWMODE) - Immediate execution

### Number Parsing

Parse priority:
1. **Vector/Matrix** - If starts with `[` and ends with `]`
   - Uses `ast.literal_eval()` for safe parsing
2. **Integer** - If no `.`, `e`, or `j`
3. **Float/Complex** - Parse as complex, return real if imag≈0

Example:
```python
"3"       → int(3)
"3.14"    → float(3.14)
"3+4j"    → complex(3+4j)
"[1,2,3]" → list([1, 2, 3])
```

---

## Error Handling

### Error Philosophy

1. **Preserve Stack**: On error, restore stack to pre-operation state
2. **Clear Messages**: Tell user what went wrong
3. **No Crashes**: Catch and report errors gracefully
4. **Continue Execution**: Error doesn't terminate calculator

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

1. Operation detects error condition
2. Restores operands to stack
3. Raises `CalculatorError` with descriptive message
4. Caught in `process_tokens`
5. Error message printed to console
6. Execution continues with next command

### Error Examples

```
> [1,2,3] CONV
Error: CONV needs 2 vectors
Stack: [1, 2, 3]  (preserved)

> 5 0 /
Error: Divide by zero
Stack: [5, 0]  (preserved)

> [[1,2]] DET
Error: DET requires square matrix
Stack: [[1, 2]]  (preserved)
```

---

## Type System

### Supported Types

1. **Scalar Numbers**
   - `int` - Integers (exact arithmetic when possible)
   - `float` - Real numbers (double precision)
   - `complex` - Complex numbers (a+bj notation)

2. **Vectors**
   - Python `list` or `tuple` of numbers
   - Must be 1D (no nested lists)
   - Example: `[1, 2, 3]`
   - Can contain int, float, or complex

3. **Matrices**
   - Python `list` of lists
   - All rows must have same length
   - Example: `[[1,2], [3,4]]`
   - Converted to NumPy for operations

### Type Detection

```python
def is_vector(x):
    """Check if x is a 1D vector"""
    return isinstance(x, (list, tuple)) and x and \
           not isinstance(x[0], (list, tuple))

def is_matrix(x):
    """Check if x is a 2D matrix"""
    return isinstance(x, (list, tuple)) and x and \
           isinstance(x[0], (list, tuple)) and \
           all(len(row) == len(x[0]) for row in x)
```

### Type Coercion

**Vector Operations:**
- Scalar + Vector → broadcast scalar to each element
- Vector + Vector → element-wise operation
- Vectors must have matching lengths

**Matrix Operations:**
- Automatic NumPy conversion for efficiency
- Results converted back to Python lists
- Complex → real when imaginary part < 1e-10
- Float → int when value is integer (within 1e-10)

### Type Preservation

The calculator attempts to preserve types:
- Integer operations return integers when possible
- Real operations on real inputs return real results
- Complex results only when mathematically necessary

---

## FFT Implementation

### Zero-Padding Strategy

FFT requires power-of-2 length for optimal efficiency:

```python
def _next_power_of_2(n):
    """Find next power of 2 >= n"""
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

Example:
```
> [1,2,3,4,5] FFT
FFT: Zero-padded from 5 to 8 samples
```

### FFT Operations

**FFT (Forward Transform):**
- Input: Real or complex vector
- Output: Complex frequency domain vector
- Length: Same as (padded) input
- DC component at index 0
- Negative frequencies in second half

**IFFT (Inverse Transform):**
- Input: Complex frequency domain
- Output: Time/spatial domain
- Returns real values if imaginary parts < 1e-10
- Normalizes by N automatically

**FFT_MAG (Magnitude Spectrum):**
- Computes |FFT(x)| for each bin
- Useful for frequency analysis
- Always returns real values
- Symmetric for real inputs

**FFT_PHASE (Phase Spectrum):**
- Computes angle(FFT(x)) for each bin
- Respects calculator's angle mode (DEG/RAD)
- Returns wrapped phase [-π,π] or [-180°,180°]

### Conjugate Symmetry

For real inputs, FFT exhibits conjugate symmetry:
- FFT[k] = conj(FFT[N-k])
- Only first N/2+1 bins needed for real signals
- Full output provided for consistency

---

## Signal Processing

### Convolution Theory

**1D Convolution (CONV)**

Discrete convolution is defined as:
```
c[n] = Σ a[k] · b[n-k]
      k
```

Properties:
- Result length: len(a) + len(b) - 1
- Commutative: a⊗b = b⊗a
- Associative: (a⊗b)⊗c = a⊗(b⊗c)
- Distributive: a⊗(b+c) = a⊗b + a⊗c

Implementation uses NumPy's `convolve()` with `mode='full'`.

**Applications:**
1. **Filtering**: Apply FIR filters to signals
   ```
   [signal] [0.333,0.333,0.333] CONV  # 3-point average
   ```

2. **Smoothing**: Reduce noise
   ```
   [noisy_data] [0.5,0.5] CONV  # 2-point average
   ```

3. **Edge Detection**: Find discontinuities
   ```
   [signal] [1,-1] CONV  # Discrete derivative
   ```

4. **Polynomial Multiplication**: Convolve coefficients
   ```
   [1,2,1] [1,3,2] CONV  # (x²+2x+1)(x²+3x+2)
   ```

**2D Convolution (CONV2)**

2D discrete convolution:
```
c[m,n] = ΣΣ a[i,j] · b[m-i,n-j]
         i j
```

Properties:
- Output size: (m₁+m₂-1) × (n₁+n₂-1)
- Used for image processing
- Kernel typically small (3×3, 5×5)

Implementation uses SciPy's `signal.convolve2d()`.

**Common Kernels:**

*Blur (3×3 box):*
```
[[1,1,1],
 [1,1,1],
 [1,1,1]]  ÷ 9
```

*Gaussian approximation:*
```
[[1,2,1],
 [2,4,2],
 [1,2,1]]  ÷ 16
```

*Sharpen:*
```
[[ 0,-1, 0],
 [-1, 5,-1],
 [ 0,-1, 0]]
```

*Sobel edge detection (horizontal):*
```
[[-1, 0, 1],
 [-2, 0, 2],
 [-1, 0, 1]]
```

*Sobel edge detection (vertical):*
```
[[-1,-2,-1],
 [ 0, 0, 0],
 [ 1, 2, 1]]
```

### Deconvolution Theory

**DECONV**

Deconvolution attempts to invert the convolution operation:
```
If c = a ⊗ b, then a = c ⊘ b
```

Implementation uses NumPy's `polydiv()` (polynomial division).

**Properties:**
- Inverse operation of convolution
- Sensitive to noise
- Works best with exact known signals
- May produce remainder (inexact recovery)

**Applications:**

1. **Signal Recovery**: Remove known filtering
   ```
   [filtered_signal] [filter] DECONV
   ```

2. **System Identification**: Find impulse response
   ```
   [output] [input] DECONV  → system response
   ```

3. **Deblurring**: Restore signals degraded by convolution

4. **Polynomial Division**: Divide polynomials
   ```
   [1,5,10,7,2] [1,2,1] DECONV  # (x⁴+5x³+...) ÷ (x²+2x+1)
   ```

**Warnings:**

The calculator warns when deconvolution is approximate:
```
> [convolved] [kernel] DECONV
Warning: Non-zero remainder (max: 1.234e-06)
```

### Cross-Correlation Theory

**XCORR**

Cross-correlation measures similarity between signals:
```
r[n] = Σ a[k] · conj(b[k+n])
       k
```

Differences from convolution:
- No time-reversal of second signal
- Used for pattern matching, not filtering
- Auto-correlation: signal correlated with itself

Implementation uses NumPy's `correlate()` with `mode='full'`.

**Applications:**

1. **Pattern Matching**: Find where pattern occurs
   ```
   [long_signal] [pattern] XCORR
   ```

2. **Time-Delay Estimation**: Measure signal lag
   ```
   [signal_1] [signal_2] XCORR
   ```

3. **Auto-Correlation**: Detect periodicities
   ```
   [signal] [signal] XCORR
   ```

4. **Signal Similarity**: Quantify how similar signals are

**Properties:**
- Peak indicates best match location
- Auto-correlation is symmetric
- Maximum at zero lag for auto-correlation

### Computational Efficiency

All signal processing operations leverage optimized libraries:
- **NumPy**: Written in C, highly optimized
- **SciPy**: Uses FFTPACK and LAPACK
- **Complexity**:
  - CONV: O(mn) direct, O((m+n)log(m+n)) via FFT
  - CONV2: O(m₁m₂n₁n₂) direct
  - DECONV: O(n²) polynomial division
  - XCORR: O(mn)

For large signals, consider FFT-based convolution (not currently implemented).

---

## Logging System

### Log File Structure

```
logs/rpn_log_YYYYMMDD_HHMMSS.txt
```

Example filename: `rpn_log_20260110_143022.txt`

### Log Entry Format

```
[HH:MM:SS] Input: <command>
  Stack before: [items]
  Stack after: [items]
  Result: <top item>

[HH:MM:SS] Input: <command>
  ERROR: <error message>
```

### Logging Flow

1. User enters command
2. Stack snapshot taken (deep copy)
3. Command executed
4. Logger records: input, before state, after state, errors
5. Entry written to file immediately (no buffering)

### Log File Lifecycle

- **Created**: When Calculator initialized with `enable_logging=True`
- **Header**: Session start timestamp
- **Written**: After each command (real-time logging)
- **Footer**: Session end timestamp and duration
- **Closed**: When calculator exits (in finally block)

### Log Control

Users can control logging:
```
> LOG ON     # Enable logging
> LOG OFF    # Disable logging
> LOG        # Show current status
```

### Logging for Library Use

When using calculator as a library, disable logging:
```python
calc = Calculator(enable_logging=False)
```

This prevents log files from being created during automated use.

---

## Extension Points

### Adding New Operations

**1. Create Module**

```python
# rpn_calculator/my_operations.py
from .errors import CalculatorError
from .utils import is_vector

def register_operations(calc):
    """Register operations with calculator"""
    return {
        "MYOP": lambda: _my_op(calc),
        "ANOTHER": lambda: _another_op(calc),
    }

def _my_op(calc):
    """My custom operation"""
    if not calc.stack:
        raise CalculatorError("MYOP needs a value")
    
    x = calc.pop()
    
    try:
        result = x * 2  # Your operation here
        calc.push(result)
    except Exception as e:
        calc.push(x)
        raise CalculatorError(f"Error in MYOP: {e}")

def _another_op(calc):
    """Another operation"""
    # Implementation here
    pass
```

**2. Register in Core**

```python
# core.py
from . import my_operations

def _register_all_operations(self):
    modules = [
        stack_operations,
        arithmetic,
        # ... existing modules
        my_operations,  # Add your module
    ]
    
    for module in modules:
        ops = module.register_operations(self)
        self.operations.update(ops)
```

**3. Add Help**

```python
# help_text.py
HELP_SECTIONS["MyOps"] = """
═══════════════════════════════════════════════════════════════════════════
MY CUSTOM OPERATIONS
═══════════════════════════════════════════════════════════════════════════
Operations:
  MYOP      Double the value on stack
  ANOTHER   Another operation

Examples:
  5 MYOP    → 10
"""
```

**4. Add Tests**

```python
# tests/test_my_operations.py
import unittest
from rpn_calculator import Calculator

class TestMyOperations(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator(enable_logging=False)
    
    def test_myop(self):
        """Test MYOP doubles value"""
        result = self.calc.evaluate_and_clear("5 MYOP")
        self.assertEqual(result, 10)
    
    def test_myop_error(self):
        """Test MYOP with empty stack"""
        self.calc.evaluate("MYOP")
        # Stack should be empty (error was handled)
        self.assertIsNone(self.calc.get_result())

if __name__ == "__main__":
    unittest.main()
```

### Custom Number Types

To add new numeric types:

1. Extend type detection in `utils.py`
2. Add handling in operation modules
3. Update `ValueFormatter.format_value()` in `formatting.py`
4. Add parsing in `_parse_and_push()` in `core.py`
5. Update type documentation

### Custom Display Formats

To add new display formats:

1. Add format option to `CalculatorState`
2. Update `ValueFormatter.format_value()` to handle new format
3. Add FORMAT command handler in `cli.py`
4. Add help text in `help_text.py`

---

## Performance Considerations

### Memory

- **Deep copies for history**: Can be expensive for large matrices
- **History unlimited**: Consider adding size limit for very long sessions
- **NumPy arrays**: Efficient storage but conversion overhead

### Computation

- **NumPy operations**: Highly optimized C code (fast)
- **SciPy functions**: LAPACK/BLAS underneath (very fast)
- **FFT**: O(N log N) - efficient even for large vectors
- **Convolution**: O(mn) direct convolution
- **Matrix operations**: O(n³) for many operations

### Optimization Opportunities

1. **History Management**: 
   - Limit history depth (e.g., last 100 operations)
   - Implement circular buffer

2. **Matrix Storage**: 
   - Keep as NumPy arrays internally
   - Convert only for display

3. **Lazy Evaluation**: 
   - Defer computation until result needed
   - Implement expression trees

4. **Caching**: 
   - Cache FFT results for repeated operations
   - Memoize expensive computations

5. **FFT-based Convolution**:
   - For large signals, use FFT method
   - Faster when len(a) + len(b) > ~100

---

## Security Considerations

### Input Validation

- All user input is tokenized and parsed safely
- No `eval()` of arbitrary Python code
- `ast.literal_eval()` used for safe list parsing (only literals)
- Malformed input caught and reported as errors

### File Operations

- Logs written to dedicated `logs/` directory
- Timestamped filenames prevent collisions
- No user-controlled file paths
- Directory created if doesn't exist

### Error Messages

- Don't expose internal paths or system info
- User-friendly messages only
- Stack traces not shown to users

### Resource Limits

Currently no limits on:
- Stack size
- Matrix dimensions
- Vector lengths
- History depth

Consider adding limits for production use.

---

## Future Enhancements

### Potential Features

1. **Enhanced Signal Processing**:
   - Windowing functions (Hamming, Hann, Blackman)
   - Filter design (Butterworth, Chebyshev)
   - Wavelet transforms
   - Z-transform operations
   - Spectrogram generation

2. **Programming Features**:
   - Scripting: Load and execute command files
   - Variables: Store and recall values by name
   - Functions: User-defined operations
   - Conditionals: If/then logic
   - Loops: Repeat operations

3. **Data Management**:
   - Import/export CSV, JSON
   - Database integration
   - Persist session state
   - Load/save workspace

4. **Visualization**:
   - Plotting integration (matplotlib)
   - Real-time graphing
   - Matrix visualization
   - Signal waveforms

5. **Advanced Math**:
   - Symbolic math (SymPy integration)
   - Units and dimensions
   - Arbitrary precision arithmetic
   - Differential equations

6. **User Interface**:
   - GUI version
   - Web interface
   - Mobile app
   - Syntax highlighting

### Plugin System

Future architecture could support plugins:

```python
class Plugin:
    """Base class for calculator plugins"""
    
    def register_operations(self, calc):
        """Return dict of operations"""
        return {}
    
    def register_help(self):
        """Return dict of help sections"""
        return {}
    
    def on_startup(self, calc):
        """Called when plugin loaded"""
        pass
    
    def on_shutdown(self, calc):
        """Called when plugin unloaded"""
        pass
```

Plugins loaded from `plugins/` directory at startup.

---

## References

### Mathematical Foundations

- **RPN**: Reverse Polish Notation (postfix notation)
- **FFT**: Cooley-Tukey algorithm, O(N log N)
- **Convolution**: Discrete linear convolution
- **Matrix Decompositions**: Standard linear algebra techniques
- **Digital Signal Processing**: Oppenheim & Schafer

### Implementation References

- NumPy documentation: https://numpy.org/doc/
- SciPy documentation: https://docs.scipy.org/
- Python typing: https://docs.python.org/3/library/typing.html
- Digital Signal Processing: https://en.wikipedia.org/wiki/Convolution

### HP Calculator Inspiration

This calculator is inspired by HP RPN calculators, particularly:
- HP-15C (scientific functions)
- HP-48 series (complex, matrices, programming)
- HP-50g (advanced features, CAS)

---

## Appendix: Code Conventions

### Naming

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private methods**: `_leading_underscore`
- **Internal helpers**: `_two_leading_underscores`

### Module Structure

```python
# 1. Module docstring
"""Module description and purpose"""

# 2. Imports (standard, third-party, local)
import math
import numpy as np
from .errors import CalculatorError

# 3. register_operations() function
def register_operations(calc):
    return {...}

# 4. Operation implementations (_op_name functions)
def _op_conv(calc):
    """Implementation"""
    pass

# 5. Helper functions
def _helper_function():
    """Helper"""
    pass

# 6. Examples and documentation (in comments/docstrings)
```

### Error Handling Pattern

```python
def _op_example(calc):
    """Standard error handling pattern"""
    # Check prerequisites
    if len(calc.stack) < 2:
        raise CalculatorError("Need 2 operands")
    
    # Pop operands
    b = calc.pop()
    a = calc.pop()
    
    # Perform operation with error recovery
    try:
        result = complex_operation(a, b)
        calc.push(result)
    except Exception as e:
        # CRITICAL: Restore stack before raising
        calc.push(a)
        calc.push(b)
        raise CalculatorError(f"Error in EXAMPLE: {e}")
```

### Documentation

- All modules have docstrings
- All functions have docstrings
- Complex algorithms have inline comments
- Help text provided for all user-facing operations
- Examples included in help and docstrings

---

## Appendix: Testing Strategy

### Test Organization

```
tests/
├── test_arithmetic.py           # Basic operations
├── test_trigonometry.py         # Trig functions
├── test_complex_numbers.py      # Complex ops
├── test_vectors.py              # Vector operations
├── test_matrices.py             # Matrix operations
├── test_fft_operations.py       # FFT tests
├── test_signal_processing.py    # Convolution tests
└── run_all_tests.py             # Test runner
```

### Test Coverage

Each test module should cover:
- **Normal operation**: Expected use cases
- **Edge cases**: Boundary conditions
- **Error handling**: Invalid inputs
- **Type handling**: Different input types
- **Integration**: Operations used together

### Test Patterns

```python
class TestOperation(unittest.TestCase):
    def setUp(self):
        """Create fresh calculator for each test"""
        self.calc = Calculator(enable_logging=False)
    
    def test_basic_operation(self):
        """Test normal operation"""
        result = self.calc.evaluate_and_clear("3 4 +")
        self.assertEqual(result, 7)
    
    def test_error_handling(self):
        """Test error preserves stack"""
        self.calc.evaluate("5 0 /")
        # Stack should be preserved
        self.assertEqual(len(self.calc.stack), 2)
```

---

*Last Updated: January 2026*