# ============================================================================
# FILE: tests/test_trigonometry.py
# ============================================================================
"""Tests for trigonometric operations"""

import unittest
import sys
import os
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpn_calculator import Calculator


class TestTrigonometry(unittest.TestCase):
    """Test trigonometric operations"""
    
    def setUp(self):
        self.calc = Calculator()
        self.calc.state.degrees = True
    
    def test_sin_degrees(self):
        """Test sine in degrees"""
        result = self.calc.evaluate_and_clear("30 SIN")
        self.assertAlmostEqual(result, 0.5, places=5)
    
    def test_cos_degrees(self):
        """Test cosine in degrees"""
        result = self.calc.evaluate_and_clear("60 COS")
        self.assertAlmostEqual(result, 0.5, places=5)
    
    def test_tan_degrees(self):
        """Test tangent in degrees"""
        result = self.calc.evaluate_and_clear("45 TAN")
        self.assertAlmostEqual(result, 1.0, places=5)
    
    def test_asin_degrees(self):
        """Test arcsine in degrees"""
        result = self.calc.evaluate_and_clear("0.5 ASIN")
        self.assertAlmostEqual(result, 30.0, places=5)
    
    def test_radians_mode(self):
        """Test trigonometry in radians mode"""
        self.calc.state.degrees = False
        result = self.calc.evaluate_and_clear(f"{math.pi/6} SIN")
        self.assertAlmostEqual(result, 0.5, places=5)


if __name__ == '__main__':
    unittest.main()

