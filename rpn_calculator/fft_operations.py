# ============================================================================
# FILE: rpn_calculator/fft_operations.py
# ============================================================================
"""FFT and spectral analysis operations"""

import math
import numpy as np
from .errors import CalculatorError

def register_operations(calc):
    """Register FFT operations"""
    return {
        "FFT": lambda: _op_fft(calc),
        "IFFT": lambda: _op_ifft(calc),
        "FFT_MAG": lambda: _op_fft_mag(calc),
        "FFT_PHASE": lambda: _op_fft_phase(calc),
    }

def _next_power_of_2(n):
    """Find the next power of 2 greater than or equal to n"""
    if n <= 0:
        return 1
    # Check if already a power of 2
    if n & (n - 1) == 0:
        return n
    # Find next power of 2
    power = 1
    while power < n:
        power *= 2
    return power

def _zero_pad_to_power_of_2(data):
    """Zero-pad data to next power of 2 length"""
    current_length = len(data)
    target_length = _next_power_of_2(current_length)
    
    if target_length == current_length:
        return data, current_length, False
    
    # Pad with zeros
    padded = list(data) + [0] * (target_length - current_length)
    return padded, current_length, True

def _op_fft(calc):
    """
    Compute Fast Fourier Transform
    Automatically zero-pads to next power of 2 if needed
    """
    if not calc.stack:
        raise CalculatorError("FFT needs a vector")
    
    data = calc.pop()
    
    if not isinstance(data, (list, tuple)):
        calc.push(data)
        raise CalculatorError("FFT expects a vector")
    
    try:
        if len(data) == 0:
            raise ValueError("Empty vector")
        
        # Zero-pad to next power of 2
        padded_data, original_length, was_padded = _zero_pad_to_power_of_2(data)
        
        # Convert to complex if not already
        data_array = np.array(padded_data, dtype=complex)
        
        # Compute FFT
        fft_result = np.fft.fft(data_array)
        
        # Convert to Python list of complex numbers
        result = fft_result.tolist()
        
        calc.push(result)
        
        # Print info
        if was_padded:
            print(f"FFT: Zero-padded from {original_length} to {len(padded_data)} samples")
        else:
            print(f"FFT: Computed on {len(data)} samples")
        
    except Exception as e:
        calc.push(data)
        raise CalculatorError(f"Error in FFT: {e}")

def _op_ifft(calc):
    """
    Compute Inverse Fast Fourier Transform
    Returns real-valued result if imaginary parts are negligible
    """
    if not calc.stack:
        raise CalculatorError("IFFT needs a vector")
    
    data = calc.pop()
    
    if not isinstance(data, (list, tuple)):
        calc.push(data)
        raise CalculatorError("IFFT expects a vector")
    
    try:
        if len(data) == 0:
            raise ValueError("Empty vector")
        
        # Zero-pad to next power of 2
        padded_data, original_length, was_padded = _zero_pad_to_power_of_2(data)
        
        # Convert to complex
        data_array = np.array(padded_data, dtype=complex)
        
        # Compute IFFT
        ifft_result = np.fft.ifft(data_array)
        
        # Check if result is essentially real
        max_imag = np.max(np.abs(ifft_result.imag))
        if max_imag < 1e-10:
            # Return real values only
            result = ifft_result.real.tolist()
        else:
            # Return complex values
            result = ifft_result.tolist()
        
        calc.push(result)
        
        if was_padded:
            print(f"IFFT: Zero-padded from {original_length} to {len(padded_data)} samples")
        else:
            print(f"IFFT: Computed on {len(data)} samples")
        
    except Exception as e:
        calc.push(data)
        raise CalculatorError(f"Error in IFFT: {e}")

def _op_fft_mag(calc):
    """
    Compute magnitude spectrum of FFT
    Returns magnitude of each frequency bin
    """
    if not calc.stack:
        raise CalculatorError("FFT_MAG needs a vector")
    
    data = calc.pop()
    
    if not isinstance(data, (list, tuple)):
        calc.push(data)
        raise CalculatorError("FFT_MAG expects a vector")
    
    try:
        if len(data) == 0:
            raise ValueError("Empty vector")
        
        # Zero-pad to next power of 2
        padded_data, original_length, was_padded = _zero_pad_to_power_of_2(data)
        
        # Convert to complex
        data_array = np.array(padded_data, dtype=complex)
        
        # Compute FFT
        fft_result = np.fft.fft(data_array)
        
        # Compute magnitude
        magnitude = np.abs(fft_result)
        
        # Convert to list
        result = magnitude.tolist()
        
        calc.push(result)
        
        if was_padded:
            print(f"FFT_MAG: Zero-padded from {original_length} to {len(padded_data)} samples")
        else:
            print(f"FFT_MAG: Computed on {len(data)} samples")
        print(f"Magnitude spectrum: {len(result)} frequency bins")
        
    except Exception as e:
        calc.push(data)
        raise CalculatorError(f"Error in FFT_MAG: {e}")

def _op_fft_phase(calc):
    """
    Compute phase spectrum of FFT
    Returns phase of each frequency bin in degrees (or radians if RAD mode)
    """
    if not calc.stack:
        raise CalculatorError("FFT_PHASE needs a vector")
    
    data = calc.pop()
    
    if not isinstance(data, (list, tuple)):
        calc.push(data)
        raise CalculatorError("FFT_PHASE expects a vector")
    
    try:
        if len(data) == 0:
            raise ValueError("Empty vector")
        
        # Zero-pad to next power of 2
        padded_data, original_length, was_padded = _zero_pad_to_power_of_2(data)
        
        # Convert to complex
        data_array = np.array(padded_data, dtype=complex)
        
        # Compute FFT
        fft_result = np.fft.fft(data_array)
        
        # Compute phase
        phase = np.angle(fft_result)
        
        # Convert to degrees if in degree mode
        if calc.state.degrees:
            phase = np.degrees(phase)
        
        # Convert to list
        result = phase.tolist()
        
        calc.push(result)
        
        if was_padded:
            print(f"FFT_PHASE: Zero-padded from {original_length} to {len(padded_data)} samples")
        else:
            print(f"FFT_PHASE: Computed on {len(data)} samples")
        
        units = "degrees" if calc.state.degrees else "radians"
        print(f"Phase spectrum: {len(result)} frequency bins ({units})")
        
    except Exception as e:
        calc.push(data)
        raise CalculatorError(f"Error in FFT_PHASE: {e}")
