# RPN Calculator

A powerful, modular scientific RPN (Reverse Polish Notation) calculator with support for complex numbers, vectors, matrices, and signal processing.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

### Core Functionality
- **RPN Entry**: Classic stack-based calculation
- **Complex Numbers**: Full support for complex arithmetic and transformations
- **Vectors**: Dot product, cross product, magnitude, normalization
- **Matrices**: Full linear algebra support including decompositions
- **FFT**: Fast Fourier Transform for signal processing
- **Session Logging**: Automatic timestamped logs of all operations

### Mathematical Operations
- Basic arithmetic (+, -, *, /, ^, MOD)
- Trigonometric functions (SIN, COS, TAN, and inverses)
- Logarithmic and exponential functions
- Integer operations (GCD, LCM, fraction conversion)
- Statistics (combinations, permutations, standard deviation)

### Matrix Operations
- Creation and manipulation
- Arithmetic (addition, multiplication, scaling)
- Properties (determinant, trace, rank, condition number)
- Advanced (eigenvalues, eigenvectors, RREF, system solving)
- Decompositions (LU, QR, SVD, Cholesky, Schur, Hessenberg)

### Signal Processing
- FFT/IFFT with automatic zero-padding
- Magnitude and phase spectrum analysis
- Supports both degrees and radians modes

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/rpn-calculator.git
cd rpn-calculator

# Install dependencies
pip install -r requirements.txt

# Run the calculator
python run_calculator.py
```

### Alternative: Install as Package

```bash
pip install -e .
```

## Quick Start

```bash
$ python run_calculator.py

> 3 4 +
Stack: [7]

> [1,2,3] [4,5,6] DOT
Stack: [32]

> [[1,2],[3,4]] DET
Stack: [-2]

> [1,2,3,4,5,6,7,8] FFT_MAG
Stack: [36, 10.46, 5.66, 4.24, 4, 4.24, 5.66, 10.46]

> @
Goodbye.
```

## Usage Examples

### Basic Arithmetic
```
> 3 4 +
7

> 10 3 - 2 *
14
```

### Complex Numbers
```
> 3 4 CMPLX ABS
5

> 1 45 RECT
0.707+0.707j
```

### Vectors
```
> [1,2,3] [4,5,6] DOT
32

> [1,0,0] [0,1,0] VCROSS
[0, 0, 1]
```

### Matrices
```
> [1,2] [3,4] 2 MATRIX
> DET
-2

> [[2,1],[1,3]] [5,6] MSOLVE
[1.8, 1.4]
```

### Signal Processing
```
> [1,2,3,4,5,6,7,8] FFT_MAG
[36, 10.46, 5.66, 4.24, 4, 4.24, 5.66, 10.46]

> [signal] FFT IFFT
(recovers original signal)
```

## Documentation

### Help System
The calculator includes a comprehensive built-in help system:

```
> HELP              # Show help menu
> HELP matrix       # Show matrix operations
> HELP fft          # Show FFT operations
> HELP SEARCH det   # Search for commands
```

### Available Commands

**Stack Operations**: C, DEL, UNDO, SWAP, RD, RU  
**Arithmetic**: +, -, *, /, ^, MOD, ||, GCD, LCM  
**Trigonometry**: SIN, COS, TAN, ASIN, ACOS, ATAN  
**Complex**: CMPLX, RECT, POLAR, RE, IM, ABS, ARG, CONJ  
**Vectors**: DOT, VMAG, VCROSS, VNORM  
**Matrices**: MATRIX, IDENTITY, DET, TRACE, MINV, M+, M-, M*  
**Decompositions**: LU, QR, SVD, CHOLESKY, SCHUR, HESSENBERG  
**FFT**: FFT, IFFT, FFT_MAG, FFT_PHASE  
**Constants**: E, PI, I  
**Modes**: DEG, RAD, DIGITS, FORMAT

See full documentation in [THEORY_OF_OPERATION.md](THEORY_OF_OPERATION.md)

## Session Logging

All calculations are automatically logged to timestamped files:

```
logs/rpn_log_20260101_123456.txt
```

Each log includes:
- Input commands
- Stack state before and after
- Results and errors
- Session duration

## Architecture

The calculator uses a modular architecture for easy maintenance and extension:

```
rpn_calculator/
├── core.py                    # Main calculator engine
├── stack_operations.py        # Stack manipulation
├── arithmetic.py              # Basic math operations
├── trigonometry.py            # Trig functions
├── complex_numbers.py         # Complex number ops
├── vectors.py                 # Vector operations
├── matrices.py                # Matrix operations
├── matrix_decompositions.py  # Matrix decompositions
├── fft_operations.py         # FFT and signal processing
├── statistics.py             # Statistical functions
├── integer_ops.py            # Integer operations
├── calc_constants.py         # Mathematical constants
├── logging.py                # Session logging
├── formatting.py             # Output formatting
├── utils.py                  # Utility functions
├── errors.py                 # Error handling
├── help_text.py              # Help documentation
└── cli.py                    # Command-line interface
```

## Testing

Run the test suite:

```bash
python tests/run_all_tests.py
```

Run individual test modules:

```bash
python tests/test_arithmetic.py
python tests/test_matrices.py
python tests/test_fft_operations.py
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development

### Adding New Operations

1. Create a new module in `rpn_calculator/`
2. Implement `register_operations(calc)` function
3. Add module to imports in `core.py`
4. Add to modules list in `_register_all_operations()`
5. Add help text to `help_text.py`
6. Write tests in `tests/`

Example:

```python
# rpn_calculator/my_operations.py
def register_operations(calc):
    return {
        "MYOP": lambda: _my_operation(calc),
    }

def _my_operation(calc):
    if not calc.stack:
        raise CalculatorError("MYOP needs a value")
    x = calc.pop()
    result = x * 2  # Your operation here
    calc.push(result)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NumPy for numerical operations
- SciPy for advanced decompositions
- Inspired by HP RPN calculators

## Authors

- Your Name (@yourusername)

## Support

For bugs and feature requests, please open an issue on GitHub.

## Changelog

### Version 1.0.0 (2026-01-01)
- Initial release
- Full RPN calculator functionality
- Complex numbers, vectors, matrices
- FFT operations
- Session logging
- Comprehensive help system
"""
