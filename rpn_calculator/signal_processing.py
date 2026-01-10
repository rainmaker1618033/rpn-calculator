# ============================================================================
# FILE: rpn_calculator/signal_processing.py
# ============================================================================
"""Signal processing operations including convolution"""

import numpy as np
from scipy import signal
from .errors import CalculatorError
from .utils import is_vector, is_matrix

def register_operations(calc):
    """Register signal processing operations"""
    return {
        "CONV": lambda: _op_conv(calc),
        "CONV2": lambda: _op_conv2(calc),
        "DECONV": lambda: _op_deconv(calc),
        "XCORR": lambda: _op_xcorr(calc),
        "CONVOLVE": lambda: _op_conv(calc),  # Alias
    }

def _op_conv(calc):
    """
    Compute convolution of two vectors
    
    Usage: vector1 vector2 CONV
    
    Computes the discrete convolution:
    c[n] = Σ a[k] * b[n-k]
    
    Result length = len(a) + len(b) - 1
    
    Examples:
        [1,2,3] [1,1,1] CONV  → [1, 3, 6, 5, 3]
        [1,1,1] [1,2,3] CONV  → [1, 3, 6, 5, 3]
    """
    if len(calc.stack) < 2:
        raise CalculatorError("CONV needs 2 vectors")
    
    b = calc.pop()
    a = calc.pop()
    
    # Check that both are vectors
    if not (is_vector(a) and is_vector(b)):
        calc.push(a)
        calc.push(b)
        raise CalculatorError("CONV requires two vectors")
    
    try:
        # Convert to lists if needed
        a_list = list(a) if not isinstance(a, list) else a
        b_list = list(b) if not isinstance(b, list) else b
        
        if len(a_list) == 0 or len(b_list) == 0:
            raise ValueError("Cannot convolve empty vectors")
        
        # Perform convolution using NumPy
        # mode='full' gives the full discrete linear convolution
        result = np.convolve(a_list, b_list, mode='full')
        
        # Convert back to Python list, handling complex numbers
        result_list = []
        for val in result:
            if isinstance(val, complex):
                if abs(val.imag) < 1e-10:
                    val = val.real
            if isinstance(val, (float, np.floating)) and abs(val - round(val)) < 1e-10:
                val = int(round(val))
            result_list.append(val)
        
        calc.push(result_list)
        
        print(f"Convolution: length {len(a_list)} * length {len(b_list)} → length {len(result_list)}")
        
    except Exception as e:
        calc.push(a)
        calc.push(b)
        raise CalculatorError(f"Error in CONV: {e}")

def _op_xcorr(calc):
    """
    Compute cross-correlation of two vectors
    
    Usage: vector1 vector2 XCORR
    
    Computes the discrete cross-correlation:
    r[n] = Σ a[k] * conj(b[k+n])
    
    Cross-correlation is similar to convolution but without time-reversal.
    Useful for signal matching and time-delay estimation.
    
    Examples:
        [1,2,3,4] [1,2,3,4] XCORR  → Auto-correlation
        [1,0,0,0] [0,0,0,1] XCORR  → Find delay
    """
    if len(calc.stack) < 2:
        raise CalculatorError("XCORR needs 2 vectors")
    
    b = calc.pop()
    a = calc.pop()
    
    if not (is_vector(a) and is_vector(b)):
        calc.push(a)
        calc.push(b)
        raise CalculatorError("XCORR requires two vectors")
    
    try:
        a_list = list(a) if not isinstance(a, list) else a
        b_list = list(b) if not isinstance(b, list) else b
        
        if len(a_list) == 0 or len(b_list) == 0:
            raise ValueError("Cannot correlate empty vectors")
        
        # Convert to numpy arrays
        a_arr = np.array(a_list)
        b_arr = np.array(b_list)
        
        # Cross-correlation using numpy
        result = np.correlate(a_arr, b_arr, mode='full')
        
        # Convert back to Python list
        result_list = []
        for val in result:
            if isinstance(val, complex):
                if abs(val.imag) < 1e-10:
                    val = val.real
            if isinstance(val, (float, np.floating)) and abs(val - round(val)) < 1e-10:
                val = int(round(val))
            result_list.append(val)
        
        calc.push(result_list)
        
        print(f"Cross-correlation: length {len(a_list)} × length {len(b_list)} → length {len(result_list)}")
        
    except Exception as e:
        calc.push(a)
        calc.push(b)
        raise CalculatorError(f"Error in XCORR: {e}")


def _op_conv2(calc):
    """
    Compute 2D convolution (for matrices/images)
    
    Usage: matrix1 matrix2 CONV2
    
    Computes 2D discrete convolution, commonly used for:
    - Image filtering
    - Edge detection in images
    - Applying kernels to 2D data
    
    The kernel (second matrix) is typically small (3x3, 5x5, etc.)
    Result size depends on 'full' mode: (m1+m2-1) × (n1+n2-1)
    
    Examples:
        # 3x3 blur kernel (box filter)
        [[1,2,1],[2,4,2],[1,2,1]] image CONV2
        
        # Sobel edge detection (horizontal)
        [[-1,0,1],[-2,0,2],[-1,0,1]] image CONV2
        
        # Simple 2x2 average
        [[0.25,0.25],[0.25,0.25]] image CONV2
    """
    if len(calc.stack) < 2:
        raise CalculatorError("CONV2 needs 2 matrices")
    
    kernel = calc.pop()
    image = calc.pop()
    
    # Check that both are matrices
    if not (is_matrix(image) and is_matrix(kernel)):
        calc.push(image)
        calc.push(kernel)
        raise CalculatorError("CONV2 requires two matrices")
    
    try:
        # Convert to numpy arrays
        img_arr = np.array(image, dtype=float)
        ker_arr = np.array(kernel, dtype=float)
        
        if img_arr.size == 0 or ker_arr.size == 0:
            raise ValueError("Cannot convolve empty matrices")
        
        # Perform 2D convolution using scipy
        result = signal.convolve2d(img_arr, ker_arr, mode='full')
        
        # Convert back to Python list of lists
        result_list = []
        for row in result:
            new_row = []
            for val in row:
                if isinstance(val, complex):
                    if abs(val.imag) < 1e-10:
                        val = val.real
                if isinstance(val, (float, np.floating)) and abs(val - round(val)) < 1e-10:
                    val = int(round(val))
                new_row.append(val)
            result_list.append(new_row)
        
        calc.push(result_list)
        
        rows_in, cols_in = img_arr.shape
        rows_ker, cols_ker = ker_arr.shape
        rows_out, cols_out = result.shape
        print(f"2D Convolution: ({rows_in}×{cols_in}) * ({rows_ker}×{cols_ker}) → ({rows_out}×{cols_out})")
        
    except ImportError:
        calc.push(image)
        calc.push(kernel)
        raise CalculatorError("CONV2 requires scipy library (pip install scipy)")
    except Exception as e:
        calc.push(image)
        calc.push(kernel)
        raise CalculatorError(f"Error in CONV2: {e}")


def _op_deconv(calc):
    """
    Compute deconvolution (inverse of convolution)
    
    Usage: convolved_signal original_signal DECONV
    
    Attempts to recover the original signal given:
    - convolved_signal: The result of a convolution
    - original_signal: One of the original signals used
    
    Deconvolution is useful for:
    - Removing blur from signals/images
    - System identification
    - Channel equalization
    - Recovering original data after filtering
    
    Note: Deconvolution is an inverse problem and can be sensitive to noise.
    Works best when you know one of the original signals exactly.
    
    Mathematical relation:
    If: c = a ⊗ b  (⊗ is convolution)
    Then: DECONV(c, b) ≈ a
    
    Examples:
        # Forward: convolve two signals
        [1,2,3] [1,1,1] CONV  → [1,3,6,5,3]
        
        # Backward: recover original
        [1,3,6,5,3] [1,1,1] DECONV  → [1,2,3]
        
        # System identification
        output input DECONV  → system_response
    """
    if len(calc.stack) < 2:
        raise CalculatorError("DECONV needs 2 vectors (convolved_signal, original_signal)")
    
    divisor = calc.pop()  # The signal to deconvolve by
    dividend = calc.pop()  # The convolved signal
    
    # Check that both are vectors
    if not (is_vector(dividend) and is_vector(divisor)):
        calc.push(dividend)
        calc.push(divisor)
        raise CalculatorError("DECONV requires two vectors")
    
    try:
        # Convert to lists
        dividend_list = list(dividend) if not isinstance(dividend, list) else dividend
        divisor_list = list(divisor) if not isinstance(divisor, list) else divisor
        
        if len(dividend_list) == 0 or len(divisor_list) == 0:
            raise ValueError("Cannot deconvolve empty vectors")
        
        # Convert to numpy arrays
        dividend_arr = np.array(dividend_list, dtype=float)
        divisor_arr = np.array(divisor_list, dtype=float)
        
        # Perform deconvolution using numpy's deconvolve
        quotient, remainder = np.polydiv(dividend_arr, divisor_arr)
        
        # Convert quotient to list
        result_list = []
        for val in quotient:
            if isinstance(val, complex):
                if abs(val.imag) < 1e-10:
                    val = val.real
            # Round near-integers
            if isinstance(val, (float, np.floating)) and abs(val - round(val)) < 1e-10:
                val = int(round(val))
            result_list.append(val)
        
        # Check if remainder is significant
        max_remainder = np.max(np.abs(remainder)) if len(remainder) > 0 else 0
        
        calc.push(result_list)
        
        print(f"Deconvolution: length {len(dividend_list)} ÷ length {len(divisor_list)} → length {len(result_list)}")
        if max_remainder > 1e-6:
            print(f"Warning: Non-zero remainder (max: {max_remainder:.6e}) - deconvolution may be approximate")
        
    except Exception as e:
        calc.push(dividend)
        calc.push(divisor)
        raise CalculatorError(f"Error in DECONV: {e}")


# ============================================================================
# EXAMPLES AND TESTS
# ============================================================================

"""
CONVOLUTION EXAMPLES:

1. Simple convolution:
   > [1,2,3] [1,1,1] CONV
   Result: [1, 3, 6, 5, 3]
   
   Explanation:
   c[0] = 1*1 = 1
   c[1] = 1*1 + 2*1 = 3
   c[2] = 1*1 + 2*1 + 3*1 = 6
   c[3] = 2*1 + 3*1 = 5
   c[4] = 3*1 = 3

2. Moving average filter:
   > [1,2,3,4,5] [0.5,0.5] CONV
   Result: [0.5, 1.5, 2.5, 3.5, 4.5, 2.5]
   
   This smooths the signal by averaging adjacent points.

3. Edge detection (discrete derivative):
   > [1,1,1,2,2,2] [1,-1] CONV
   Result: [1, 0, 0, 1, 0, 0, -2]
   
   Detects edges (changes) in the signal.

4. Impulse response:
   > [1,0,0,0] [1,2,3] CONV
   Result: [1, 2, 3, 0, 0, 0]
   
   Shows the system's response to an impulse.

5. Polynomial multiplication:
   > [1,2,1] [1,3,2] CONV
   Result: [1, 5, 10, 7, 2]
   
   (x² + 2x + 1) * (x² + 3x + 2) = x⁴ + 5x³ + 10x² + 7x + 2

6. Complex convolution:
   > [1+1j,2+2j] [1,1] CONV
   Result: [(1+1j), (3+3j), (2+2j)]


2D CONVOLUTION EXAMPLES:

1. Identity (no change):
   > [[1,2],[3,4]] [[1]] CONV2
   Result: [[1, 2], [3, 4]]

2. Simple averaging (2x2 box filter):
   > [[1,2,3],[4,5,6],[7,8,9]] [[0.25,0.25],[0.25,0.25]] CONV2
   Result: Blurred/smoothed image

3. Edge detection - Sobel horizontal:
   > image [[-1,0,1],[-2,0,2],[-1,0,1]] CONV2
   Detects vertical edges

4. Edge detection - Sobel vertical:
   > image [[-1,-2,-1],[0,0,0],[1,2,1]] CONV2
   Detects horizontal edges

5. Sharpening kernel:
   > image [[0,-1,0],[-1,5,-1],[0,-1,0]] CONV2
   Enhances edges

6. Gaussian blur (3x3 approximation):
   > image [[1,2,1],[2,4,2],[1,2,1]] CONV2
   (Note: divide by 16 for normalized kernel)


DECONVOLUTION EXAMPLES:

1. Perfect recovery:
   # Forward
   > [1,2,3] [1,1,1] CONV
   Result: [1, 3, 6, 5, 3]
   
   # Backward (recover [1,2,3])
   > [1,3,6,5,3] [1,1,1] DECONV
   Result: [1, 2, 3]

2. System identification:
   # Given output and input, find system response
   > [output_signal] [input_signal] DECONV
   Result: impulse response of system

3. Remove blur:
   # If you know the blur kernel
   > [blurred_image] [blur_kernel] DECONV
   Result: (approximately) original image

4. Polynomial division:
   # Divide polynomials
   > [1,5,10,7,2] [1,2,1] DECONV
   Result: [1, 3, 2]  (quotient)
   
   Verifies: (x² + 2x + 1) * (x² + 3x + 2) = x⁴ + 5x³ + 10x² + 7x + 2

5. Channel equalization:
   > [received_signal] [known_channel_response] DECONV
   Result: original transmitted signal


CROSS-CORRELATION EXAMPLES:

1. Auto-correlation (signal with itself):
   > [1,2,3,4] [1,2,3,4] XCORR
   Result: [4, 11, 20, 30, 20, 11, 4]
   
   Peak at center shows maximum correlation.

2. Find delay between signals:
   > [1,0,0,0,0] [0,0,1,0,0] XCORR
   Result: Shows peak at delay position

3. Pattern matching:
   > [1,2,3,2,1] [2,3,2] XCORR
   Result: [2, 7, 14, 16, 14, 7, 2]
   
   Peak shows where pattern matches best.


PRACTICAL WORKFLOWS:

1. Apply filter and verify:
   > [1,2,3,4,5] [0.5,0.5] CONV    # Apply filter
   > [0.5,0.5] DECONV               # Remove filter
   Result: Should recover [1,2,3,4,5] (approximately)

2. Image processing pipeline:
   > image [[1,2,1],[2,4,2],[1,2,1]] CONV2   # Blur
   > SOME_THRESHOLD                          # Threshold
   > edge_kernel CONV2                       # Edge detect

3. Signal analysis:
   > signal FFT                    # Transform to frequency domain
   > # analyze frequency content
   > IFFT                          # Back to time domain
   > filter CONV                   # Apply time-domain filter
"""