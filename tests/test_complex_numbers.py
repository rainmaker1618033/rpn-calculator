
# ============================================================================
# FILE: tests/test_complex_numbers.py
# ============================================================================
"""Tests for complex number operations"""

import unittest
import sys
import os
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpn_calculator import Calculator


class TestComplexNumbers(unittest.TestCase):
    """Test complex number operations"""
    
    def setUp(self):
        self.calc = Calculator()
    
    def test_cmplx_creation(self):
        """Test complex number creation"""
        result = self.calc.evaluate_and_clear("3 4 CMPLX")
        self.assertEqual(result, 3+4j)
    
    def test_abs(self):
        """Test absolute value of complex number"""
        result = self.calc.evaluate_and_clear("3 4 CMPLX ABS")
        self.assertEqual(result, 5.0)
    
    def test_real_part(self):
        """Test extracting real part"""
        result = self.calc.evaluate_and_clear("3 4 CMPLX RE")
        self.assertEqual(result, 3.0)
    
    def test_imag_part(self):
        """Test extracting imaginary part"""
        result = self.calc.evaluate_and_clear("3 4 CMPLX IM")
        self.assertEqual(result, 4.0)
    
    def test_conjugate(self):
        """Test complex conjugate"""
        result = self.calc.evaluate_and_clear("3 4 CMPLX CONJ")
        self.assertEqual(result, 3-4j)
    
    def test_rect_to_polar(self):
        """Test rectangular to polar conversion"""
        self.calc.evaluate("1 1 CMPLX POLAR")
        theta = self.calc.pop()
        r = self.calc.pop()
        self.assertAlmostEqual(r, math.sqrt(2), places=5)
        self.assertAlmostEqual(theta, 45.0, places=5)  # degrees
    
    def test_polar_to_rect(self):
        """Test polar to rectangular conversion"""
        result = self.calc.evaluate_and_clear("1 45 RECT")
        self.assertAlmostEqual(result.real, 1/math.sqrt(2), places=5)
        self.assertAlmostEqual(result.imag, 1/math.sqrt(2), places=5)


if __name__ == '__main__':
    unittest.main()

