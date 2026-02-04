# RPN Calculator Library API Documentation

## Version

Version: 1.0.0

## Overview

The RPN Calculator is a Python library that provides a powerful stack-based calculator with support for basic arithmetic, trigonometry, complex numbers, vectors, matrices, FFT operations, and more. It can be easily integrated into other applications as a computational engine.

## Installation

```python
from rpn_calculator import Calculator, CalculatorError
```

## Basic Usage

### Creating a Calculator Instance

```python
# Create calculator (logging enabled by default for CLI use)
calc = Calculator()

# Create calculator without logging (recommended for library use)
calc = Calculator(enable_logging=False)
```

### Core API Methods

#### `evaluate(expression: str) -> Any`
Evaluates an RPN expression and returns the top of the stack.

```python
calc = Calculator(enable_logging=False)
result = calc.evaluate("3 4 +")
print(result)  # 7
```

#### `evaluate_and_clear(expression: str) -> Any`
Evaluates an RPN expression, returns the result, and clears the stack.

```python
result = calc.evaluate_and_clear("5 2 *")
print(result)  # 10
# Stack is now empty
```

#### `push(value: Any)`
Pushes a value onto the stack.

```python
calc.push(10)
calc.push(20)
```

#### `pop() -> Any`
Pops and returns the top value from the stack.

```python
calc.push(42)
value = calc.pop()  # 42
```

#### `peek() -> Any`
Returns the top value without removing it.

```python
calc.push(15)
value = calc.peek()  # 15, stack still contains 15
```

#### `get_result() -> Any`
Returns the top of stack without popping (returns None if empty).

```python
calc.evaluate("7 3 -")
result = calc.get_result()  # 4
```

#### Direct Stack Access

```python
calc.stack  # Direct access to the stack list
```

## Configuration

### Calculator State

```python
calc.state.degrees = True      # Angle mode: True=degrees, False=radians
calc.state.digits = 6          # Display precision
calc.state.format = "FLOAT"    # Display format: "FLOAT" or "SCIENTIFIC"
```

## Supported Operations

### Arithmetic
- `+`, `-`, `*`, `/`, `^` (power), `MOD`
- `||` - Parallel operation: (x×y)/(x+y)

### Trigonometry
- `SIN`, `COS`, `TAN`, `ASIN`, `ACOS`, `ATAN`
- Respects degrees/radians mode

### Logarithmic
- `LOG10` (base 10), `LN` (natural log), `EXP`, `SQRT`
- `1/X` - reciprocal

### Complex Numbers
- `CMPLX` - Create complex (real, imag)
- `RECT` - Polar to rectangular (r, θ)
- `POLAR` - Rectangular to polar
- `RE`, `IM`, `ABS`, `ARG`, `CONJ`

### Vectors
- `DOT` - Dot product
- `VMAG` - Vector magnitude
- `VCROSS` - Cross product (3D vectors)
- `VNORM` - Normalize vector

### Matrices
- `MATRIX` - Create matrix from rows
- `DET`, `TRACE`, `MINV` (inverse)
- `M+`, `M-`, `M*` - Matrix operations
- `MTRANS` - Transpose

### Matrix Decompositions
- `LU`, `QR`, `SVD`, `CHOLESKY`, `SCHUR`, `EIGEN`

### FFT & Signal Processing
- `FFT`, `IFFT`, `FFT_MAG`, `FFT_PHASE`
- `CONV` - Convolution
- `CORR` - Correlation
- `FILTER` - Digital filter

### Stack Operations
- `C` - Clear stack
- `DEL` - Delete top item
- `SWAP` - Swap top two items
- `RD` - Roll down
- `RU` - Roll up
- `UNDO` - Undo last operation

### Statistics & Integers
- `COMB`, `PERM`, `STDV`
- `GCD`, `LCM`

### Constants
- `PI`, `E`, `I` (imaginary unit)

## Data Types

The calculator handles:
- **Scalars**: integers, floats, complex numbers
- **Vectors**: Python lists `[1, 2, 3]`
- **Matrices**: Lists of lists `[[1, 2], [3, 4]]`

## Error Handling

```python
from rpn_calculator import CalculatorError

try:
    result = calc.evaluate("5 0 /")
except CalculatorError as e:
    print(f"Error: {e.message}")
```

## Example: CSV Column Processing

This example shows how to use the RPN calculator as a library to process data columns from a CSV file.

```python
#!/usr/bin/env python3
"""
CSV Column Calculator
Reads a CSV file and performs RPN calculations on columns.
"""

import csv
from rpn_calculator import Calculator, CalculatorError

def process_csv(filename, operations):
    """
    Process CSV columns with RPN operations.
    
    Args:
        filename: Path to CSV file
        operations: Dict mapping output column names to RPN expressions
                   Column references use $A, $B, $C notation
    
    Example:
        operations = {
            'sum': '$A $B +',
            'product': '$A $B *',
            'average': '$A $B + 2 /'
        }
    """
    # Read CSV data
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames
    
    # Create calculator (disable logging for library use)
    calc = Calculator(enable_logging=False)
    
    # Process each row
    results = []
    for row_num, row in enumerate(rows, start=2):  # Start at 2 (header is row 1)
        row_result = dict(row)  # Copy original data
        
        for output_col, expression in operations.items():
            # Substitute column references
            expr = expression
            for header in headers:
                col_ref = f'${chr(65 + headers.index(header))}'  # $A, $B, $C...
                if col_ref in expr:
                    value = row[header]
                    expr = expr.replace(col_ref, value)
            
            try:
                # Evaluate expression
                result = calc.evaluate_and_clear(expr)
                row_result[output_col] = result
            except CalculatorError as e:
                print(f"Row {row_num}, {output_col}: Error - {e.message}")
                row_result[output_col] = None
            except Exception as e:
                print(f"Row {row_num}, {output_col}: Error - {str(e)}")
                row_result[output_col] = None
        
        results.append(row_result)
    
    return results

def write_csv(filename, data, fieldnames):
    """Write results to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# Example usage
if __name__ == "__main__":
    # Define calculations
    operations = {
        'total': '$A $B +',                    # Sum of columns A and B
        'average': '$A $B + 2 /',              # Average of A and B
        'variance': '$A $B - 2 ^',             # Squared difference
        'hypotenuse': '$A 2 ^ $B 2 ^ + SQRT',  # sqrt(A² + B²)
    }
    
    # Process CSV
    results = process_csv('input.csv', operations)
    
    # Write output
    if results:
        all_headers = list(results[0].keys())
        write_csv('output.csv', results, all_headers)
        print(f"Processed {len(results)} rows")
        print(f"Output written to: output.csv")

# Sample input.csv:
# x,y
# 3,4
# 5,12
# 8,15
#
# Output will include:
# x,y,total,average,variance,hypotenuse
# 3,4,7,3.5,1,5.0
# 5,12,17,8.5,49,13.0
# 8,15,23,11.5,49,17.0
```

## Advanced Example: Vector Operations on CSV

```python
from rpn_calculator import Calculator

def process_vector_columns(filename):
    """Process CSV rows as vectors and compute dot products."""
    import csv
    
    calc = Calculator(enable_logging=False)
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row_num, row in enumerate(reader, start=2):
            # Convert row to vector of floats
            vector = [float(x) for x in row]
            
            # Push as vector and compute magnitude
            calc.push(vector)
            calc.evaluate("VMAG")
            magnitude = calc.pop()
            
            print(f"Row {row_num} magnitude: {magnitude:.4f}")

# Example: vectors.csv contains
# 1,0,0
# 3,4,0
# 1,1,1
```

## Tips for Library Integration

1. **Disable logging** when using as a library: `Calculator(enable_logging=False)`
2. **Use `evaluate_and_clear()`** for independent calculations to avoid stack accumulation
3. **Handle CalculatorError exceptions** for robust error handling
4. **Access calculator state** to configure angle modes and display preferences
5. **Direct stack manipulation** is available via `push()`, `pop()`, and `calc.stack`

## RPN Expression Examples

```python
calc = Calculator(enable_logging=False)

# Basic arithmetic
calc.evaluate("5 3 +")           # 8
calc.evaluate("10 3 /")          # 3.333...
calc.evaluate("2 8 ^")           # 256

# Complex numbers
calc.evaluate("3 4 CMPLX ABS")   # 5.0  (magnitude of 3+4i)

# Vectors
calc.evaluate("[1,2,3] [4,5,6] DOT")  # 32 (dot product)

# Trigonometry (ensure degrees mode)
calc.state.degrees = True
calc.evaluate("30 SIN")          # 0.5

# Multi-step calculations
calc.evaluate("5 2 3 + *")       # 25  (5 * (2+3))
```



