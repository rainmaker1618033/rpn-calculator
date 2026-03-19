# RPN Calculator

A modular scientific RPN (Reverse Polish Notation) calculator with support for complex numbers, vectors, matrices, and advanced signal processing.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

### Core Functionality
- **RPN Entry**: Classic stack-based calculation
- **Complex Numbers**: Full support for complex arithmetic and transformations
- **Vectors**: Dot product, cross product, magnitude, normalization
- **Matrices**: Full linear algebra support including decompositions
- **Signal Processing**: FFT, convolution, deconvolution, and correlation
- **Session Logging**: Automatic timestamped logs of all operations

### Mathematical Operations
- Basic arithmetic (+, -, *, /, ^, MOD, ||)
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
- **FFT Operations**: FFT, IFFT with automatic zero-padding to power of 2
- **Magnitude and Phase**: FFT_MAG, FFT_PHASE for spectral analysis
- **1D Convolution**: CONV for filtering, smoothing, and polynomial multiplication
- **2D Convolution**: CONV2 for image processing and 2D filtering
- **Deconvolution**: DECONV for signal recovery and system identification
- **Cross-Correlation**: XCORR for pattern matching and delay estimation
- **Supports degrees and radians modes** for phase calculations

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/rainmaker1618033/rpn-calculator.git
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

> [1,2,3] [1,1,1] CONV
Stack: [1, 3, 6, 5, 3]

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

> 2 8 ^
256
```

### Complex Numbers
```
> 3 4 CMPLX ABS
5

> 1 45 RECT
0.707+0.707j

> 3+4j POLAR
5, 53.13 (in DEG mode)
```

### Vectors
```
> [1,2,3] [4,5,6] DOT
32

> [1,0,0] [0,1,0] VCROSS
[0, 0, 1]

> [3,4] VMAG
5
```

### Matrices
```
> [1,2] [3,4] 2 MATRIX
> DET
-2

> [[2,1],[1,3]] [5,6] MSOLVE
[1.8, 1.4]

> [[1,2],[3,4]] MINV
[[-2, 1], [1.5, -0.5]]
```

### Signal Processing - FFT
```
> [1,2,3,4,5,6,7,8] FFT_MAG
[36, 10.46, 5.66, 4.24, 4, 4.24, 5.66, 10.46]

> [signal] FFT IFFT
(recovers original signal)

> [1,1,0,0] FFT_PHASE
(phase spectrum in degrees or radians)
```

### Signal Processing - Convolution
```
# 1D Convolution (smoothing filter)
> [1,2,3,4,5] [0.5,0.5] CONV
[0.5, 1.5, 2.5, 3.5, 4.5, 2.5]

# Edge detection
> [1,1,1,2,2,2] [1,-1] CONV
[1, 0, 0, 1, 0, 0, -2]

# Polynomial multiplication: (x+1)(x+2) = x² + 3x + 2
> [1,1] [1,2] CONV
[1, 3, 2]

# 2D Convolution (image blur)
> [[1,2,3],[4,5,6],[7,8,9]] [[0.25,0.25],[0.25,0.25]] CONV2
(blurred 4×4 result)

# Edge detection (Sobel)
> image [[-1,0,1],[-2,0,2],[-1,0,1]] CONV2
(detects vertical edges)
```

### Signal Processing - Deconvolution
```
# Recover original signal
> [1,2,3] [1,1,1] CONV
[1, 3, 6, 5, 3]

> [1,1,1] DECONV
[1, 2, 3]

# Polynomial division: (x²+3x+2) ÷ (x+1) = (x+2)
> [1,3,2] [1,1] DECONV
[1, 2]
```

### Signal Processing - Cross-Correlation
```
# Auto-correlation (find periodicities)
> [1,2,3,4] [1,2,3,4] XCORR
[4, 11, 20, 30, 20, 11, 4]

# Pattern matching
> [1,2,3,2,1,0,0,1,2,3,2,1] [2,3,2] XCORR
(peak shows where pattern occurs)
```

## Documentation

### Help System
The calculator includes a comprehensive built-in help system:

```
> HELP              # Show help menu
> HELP matrix       # Show matrix operations
> HELP fft          # Show FFT operations
> HELP signal       # Show convolution operations
> HELP SEARCH conv  # Search for commands
```

### Available Commands

**Stack Operations**: C, DEL, UNDO, SWAP, RD, RU  
**Arithmetic**: +, -, *, /, ^, MOD, ||, GCD, LCM  
**Trigonometry**: SIN, COS, TAN, ASIN, ACOS, ATAN  
**Logarithmic**: LOG, LOG2, LN, EXP, SQRT, 1/X  
**Complex**: CMPLX, RECT, POLAR, RE, IM, ABS, ARG, CONJ  
**Vectors**: DOT, VMAG, VCROSS, VNORM  
**Matrices**: MATRIX, IDENTITY, DET, TRACE, MINV, M+, M-, M*, MSOLVE  
**Decompositions**: LU, QR, SVD, CHOLESKY, SCHUR, HESSENBERG  
**FFT**: FFT, IFFT, FFT_MAG, FFT_PHASE  
**Signal Processing**: CONV, CONV2, DECONV, XCORR  
**Constants**: E, PI, I  
**Modes**: DEG, RAD, DIGITS, FORMAT, LOG

See full documentation in [THEORY_OF_OPERATION.md](THEORY_OF_OPERATION.md)

## Signal Processing Applications

### 1D Convolution (CONV)
- **Filtering**: Moving average, smoothing, noise reduction
- **Edge Detection**: Find discontinuities in signals
- **Polynomial Multiplication**: Efficient polynomial operations
- **System Analysis**: Impulse response calculations

### 2D Convolution (CONV2)
- **Image Filtering**: Blur, sharpen, smooth images
- **Edge Detection**: Sobel, Prewitt, Laplacian operators
- **Feature Extraction**: Pattern recognition in images
- **Kernel Operations**: Apply custom filters to 2D data

### Deconvolution (DECONV)
- **Signal Recovery**: Remove known filtering effects
- **System Identification**: Find impulse response from input/output
- **Deblurring**: Restore signals degraded by convolution
- **Polynomial Division**: Algebraic polynomial operations

### Cross-Correlation (XCORR)
- **Pattern Matching**: Find similar patterns in signals
- **Time-Delay Estimation**: Measure signal delays
- **Signal Similarity**: Quantify signal relationships
- **Auto-Correlation**: Detect periodicities in data

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

Control logging:
```
> LOG ON          # Enable logging
> LOG OFF         # Disable logging
```

## Architecture

The calculator uses a modular architecture for easy maintenance and extension:

```
rpn_calculator/
├── core.py                    # Main calculator engine
├── stack_operations.py        # Stack manipulation
├── arithmetic.py              # Basic math operations
├── trigonometry.py            # Trig functions
├── logarithmic.py             # Log and exponential
├── complex_numbers.py         # Complex number ops
├── vectors.py                 # Vector operations
├── matrices.py                # Matrix operations
├── matrix_decompositions.py  # Matrix decompositions
├── fft_operations.py          # FFT and spectral analysis
├── signal_processing.py       # Convolution and correlation
├── statistics.py              # Statistical functions
├── integer_ops.py             # Integer operations
├── calc_constants.py          # Mathematical constants
├── logging.py                 # Session logging
├── formatting.py              # Output formatting
├── utils.py                   # Utility functions
├── errors.py                  # Error handling
├── help_text.py               # Help documentation
└── cli.py                     # Command-line interface
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
python tests/test_signal_processing.py
```

Test coverage includes:
- ✓ Arithmetic operations
- ✓ Complex numbers
- ✓ Vectors and matrices
- ✓ Matrix decompositions
- ✓ FFT operations
- ✓ Convolution and deconvolution
- ✓ Cross-correlation
- ✓ Error handling

## Common Workflows

### Signal Analysis Pipeline
```
# Load signal → FFT → Magnitude spectrum
> [1,2,3,4,5,6,7,8] FFT FFT_MAG

# Filter signal → Analyze
> [noisy_signal] [0.333,0.333,0.333] CONV FFT_MAG

# Roundtrip test
> [signal] FFT IFFT  (should recover original)
```

### Image Processing Pipeline
```
# Blur → Sharpen → Edge detect
> image [[1,1],[1,1]] CONV2
> [[0,-1,0],[-1,5,-1],[0,-1,0]] CONV2
> [[-1,0,1],[-2,0,2],[-1,0,1]] CONV2
```

### System Identification
```
# Known input and output → Find system response
> [output] [input] DECONV
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
from .errors import CalculatorError

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

## Dependencies

### Required
- **numpy** >= 1.20.0 - Numerical operations
- **scipy** >= 1.7.0 - Advanced matrix decompositions and signal processing

### Optional
- **sympy** - For symbolic calculator extension
- **matplotlib** - For plotting capabilities

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NumPy for numerical operations
- SciPy for advanced decompositions and signal processing
- Inspired by HP RPN calculators

## Authors

- G Stults (@rainmaker1618033)

## Support

For bugs and feature requests, please open an issue on GitHub.

## Changelog

### Version 1.0.0 (2026-01-10)
- Initial release
- Full RPN calculator functionality
- Complex numbers, vectors, matrices
- FFT operations with auto-padding
- 1D convolution (CONV) and deconvolution (DECONV)
- 2D convolution (CONV2) for image processing
- Cross-correlation (XCORR)
- Session logging with LOG ON/OFF
- Comprehensive help system
- 40+ unit tests for signal processing
- Graceful error handling

## Related Projects

- **Symbolic Calculator** - Extension with symbolic math and plotting (see `examples/`)

## Roadmap

Future enhancements under consideration:
- [ ] Windowing functions (Hamming, Hann, Blackman)
- [ ] Additional filters (Butterworth, Chebyshev)
- [ ] Wavelet transforms
- [ ] Z-transform operations
- [ ] Digital filter design tools
- [ ] Spectrogram visualization
- [ ] Web interface
- [ ] Plugin system for custom operations
