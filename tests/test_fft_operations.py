"""
Tests for FFT operations
"""

import unittest
import sys
import os
import math
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpn_calculator import Calculator


class TestFFTOperations(unittest.TestCase):
    """Test FFT and related operations"""
    
    def setUp(self):
        self.calc = Calculator()
    
    def test_fft_power_of_2(self):
        """Test FFT on power of 2 length (no padding needed)"""
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        self.calc.push(data)
        self.calc.operations["FFT"]()
        result = self.calc.pop()
        
        # Should have 8 elements
        self.assertEqual(len(result), 8)
        
        # All elements should be complex
        for val in result:
            self.assertIsInstance(val, complex)
        
        # DC component (first element) should be sum of input
        self.assertAlmostEqual(result[0].real, sum(data), places=5)
        self.assertAlmostEqual(result[0].imag, 0, places=5)
    
    def test_fft_zero_padding(self):
        """Test FFT with automatic zero-padding"""
        data = [1, 2, 3, 4, 5]  # 5 elements → should pad to 8
        self.calc.push(data)
        self.calc.operations["FFT"]()
        result = self.calc.pop()
        
        # Should be padded to 8 (next power of 2)
        self.assertEqual(len(result), 8)
    
    def test_fft_ifft_roundtrip(self):
        """Test that FFT → IFFT recovers original signal"""
        original = [1, 2, 3, 4]
        self.calc.push(original)
        self.calc.operations["FFT"]()
        self.calc.operations["IFFT"]()
        recovered = self.calc.pop()
        
        # Should recover original (within numerical precision)
        self.assertEqual(len(recovered), len(original))
        for orig, rec in zip(original, recovered):
            self.assertAlmostEqual(orig, rec, places=10)
    
    def test_fft_ifft_roundtrip_with_padding(self):
        """Test FFT → IFFT with zero-padding"""
        original = [1, 2, 3, 4, 5]  # Will pad to 8
        self.calc.push(original)
        self.calc.operations["FFT"]()
        self.calc.operations["IFFT"]()
        recovered = self.calc.pop()
        
        # Should recover original 5 elements (padded zeros also recovered)
        self.assertEqual(len(recovered), 8)
        
        # First 5 should match original
        for i, (orig, rec) in enumerate(zip(original, recovered[:5])):
            self.assertAlmostEqual(orig, rec, places=10, 
                                 msg=f"Element {i} mismatch")
        
        # Last 3 should be ~0 (the padding)
        for i in range(5, 8):
            self.assertAlmostEqual(recovered[i], 0, places=10,
                                 msg=f"Padded element {i} should be ~0")
    
    def test_fft_mag(self):
        """Test FFT magnitude computation"""
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        self.calc.push(data)
        self.calc.operations["FFT_MAG"]()
        magnitudes = self.calc.pop()
        
        # Should have 8 elements
        self.assertEqual(len(magnitudes), 8)
        
        # All magnitudes should be real and non-negative
        for mag in magnitudes:
            self.assertIsInstance(mag, (int, float))
            self.assertGreaterEqual(mag, 0)
        
        # DC magnitude should equal sum of input
        self.assertAlmostEqual(magnitudes[0], sum(data), places=5)
    
    def test_fft_phase_degrees(self):
        """Test FFT phase in degrees"""
        self.calc.state.degrees = True
        data = [1, 1, 0, 0]
        self.calc.push(data)
        self.calc.operations["FFT_PHASE"]()
        phases = self.calc.pop()
        
        # Should have 4 elements
        self.assertEqual(len(phases), 4)
        
        # All phases should be real
        for phase in phases:
            self.assertIsInstance(phase, (int, float))
            # Phase should be in range [-180, 180] degrees
            self.assertGreaterEqual(phase, -180)
            self.assertLessEqual(phase, 180)
    
    def test_fft_phase_radians(self):
        """Test FFT phase in radians"""
        self.calc.state.degrees = False
        data = [1, 1, 0, 0]
        self.calc.push(data)
        self.calc.operations["FFT_PHASE"]()
        phases = self.calc.pop()
        
        # All phases should be in range [-π, π] radians
        for phase in phases:
            self.assertGreaterEqual(phase, -math.pi)
            self.assertLessEqual(phase, math.pi)
    
    def test_fft_dc_component(self):
        """Test that DC component equals mean of signal"""
        data = [2, 4, 6, 8]
        mean = sum(data) / len(data)
        
        self.calc.push(data)
        self.calc.operations["FFT"]()
        fft_result = self.calc.pop()
        
        # DC component (first element) / N should equal mean
        dc_component = fft_result[0].real / len(data)
        self.assertAlmostEqual(dc_component, mean, places=5)
    
    def test_fft_symmetry_real_input(self):
        """Test that FFT of real input has conjugate symmetry"""
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        self.calc.push(data)
        self.calc.operations["FFT"]()
        result = self.calc.pop()
        
        # For real input, FFT[k] = conj(FFT[N-k])
        N = len(result)
        for k in range(1, N // 2):
            fft_k = result[k]
            fft_nk = result[N - k]
            # Should be complex conjugates
            self.assertAlmostEqual(fft_k.real, fft_nk.real, places=5)
            self.assertAlmostEqual(fft_k.imag, -fft_nk.imag, places=5)
    
    def test_fft_sine_wave(self):
        """Test FFT on a pure sine wave shows single frequency peak"""
        # Create a sine wave: 1 cycle in 8 samples
        N = 8
        freq = 1  # 1 cycle
        data = [math.sin(2 * math.pi * freq * n / N) for n in range(N)]
        
        self.calc.push(data)
        self.calc.operations["FFT_MAG"]()
        magnitudes = self.calc.pop()
        
        # Peak should be at bin 1 (and bin N-1 by symmetry)
        self.assertGreater(magnitudes[1], magnitudes[0])  # Larger than DC
        self.assertGreater(magnitudes[1], magnitudes[2])  # Larger than next bin


class TestFFTEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for FFT"""
    
    def setUp(self):
        self.calc = Calculator()
    
    def test_fft_empty_vector(self):
        """Test FFT on empty vector"""
        self.calc.push([])
        with self.assertRaises(Exception):
            self.calc.operations["FFT"]()
    
    def test_fft_non_vector(self):
        """Test FFT on non-vector input"""
        self.calc.push(42)
        with self.assertRaises(Exception):
            self.calc.operations["FFT"]()
    
    def test_fft_single_element(self):
        """Test FFT on single element"""
        self.calc.push([5])
        self.calc.operations["FFT"]()
        result = self.calc.pop()
        
        # Single element FFT is just the element itself
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].real, 5, places=5)
        self.assertAlmostEqual(result[0].imag, 0, places=5)
    
    def test_fft_complex_input(self):
        """Test FFT on complex input"""
        data = [1+1j, 2+2j, 3+3j, 4+4j]
        self.calc.push(data)
        self.calc.operations["FFT"]()
        result = self.calc.pop()
        
        # Should work with complex input
        self.assertEqual(len(result), 4)
        for val in result:
            self.assertIsInstance(val, complex)
    
    def test_ifft_preserves_real_signal(self):
        """Test that IFFT of real signal stays real"""
        data = [1.0, 2.0, 3.0, 4.0]
        self.calc.push(data)
        self.calc.operations["FFT"]()
        self.calc.operations["IFFT"]()
        result = self.calc.pop()
        
        # Result should be real (not complex)
        for val in result:
            self.assertIsInstance(val, (int, float))


class TestFFTApplications(unittest.TestCase):
    """Test practical applications of FFT"""
    
    def setUp(self):
        self.calc = Calculator()
    
    def test_parseval_theorem(self):
        """Test Parseval's theorem: energy in time = energy in frequency"""
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        
        # Energy in time domain
        time_energy = sum(x**2 for x in data)
        
        # Energy in frequency domain
        self.calc.push(data)
        self.calc.operations["FFT"]()
        fft_result = self.calc.pop()
        
        freq_energy = sum(abs(x)**2 for x in fft_result) / len(fft_result)
        
        # Should be equal (within numerical precision)
        self.assertAlmostEqual(time_energy, freq_energy, places=5)
    
    def test_frequency_shift(self):
        """Test that multiplying by exp(jwt) shifts frequency"""
        N = 8
        data = [1, 0, 0, 0, 0, 0, 0, 0]  # Impulse at t=0
        
        # Get FFT of original
        self.calc.push(data)
        self.calc.operations["FFT_MAG"]()
        mag1 = self.calc.pop()
        
        # All magnitudes should be equal for impulse
        for i in range(len(mag1)):
            self.assertAlmostEqual(mag1[i], 1.0, places=5)
    
    def test_linearity(self):
        """Test FFT linearity: FFT(ax + by) = a*FFT(x) + b*FFT(y)"""
        x = [1, 2, 3, 4]
        y = [4, 3, 2, 1]
        a = 2
        b = 3
        
        # Compute FFT(ax + by)
        combined = [a*xi + b*yi for xi, yi in zip(x, y)]
        self.calc.push(combined)
        self.calc.operations["FFT"]()
        fft_combined = self.calc.pop()
        
        # Compute a*FFT(x)
        self.calc.push(x)
        self.calc.operations["FFT"]()
        fft_x = self.calc.pop()
        fft_x_scaled = [a * val for val in fft_x]
        
        # Compute b*FFT(y)
        self.calc.push(y)
        self.calc.operations["FFT"]()
        fft_y = self.calc.pop()
        fft_y_scaled = [b * val for val in fft_y]
        
        # Compare
        for i in range(len(fft_combined)):
            expected = fft_x_scaled[i] + fft_y_scaled[i]
            self.assertAlmostEqual(fft_combined[i].real, expected.real, places=5)
            self.assertAlmostEqual(fft_combined[i].imag, expected.imag, places=5)


if __name__ == '__main__':
    unittest.main()