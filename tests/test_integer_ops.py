# ============================================================================
# FILE: tests/test_integer_ops.py
# ============================================================================
"""Tests for integer operations"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpn_calculator import Calculator


class TestIntegerOps(unittest.TestCase):
    """Test integer operations"""
    
    def setUp(self):
        self.calc = Calculator()
    
    def test_gcd(self):
        """Test greatest common divisor"""
        result = self.calc.evaluate_and_clear("12 18 GCD")
        self.assertEqual(result, 6)
    
    def test_gcd_coprime(self):
        """Test GCD of coprime numbers"""
        result = self.calc.evaluate_and_clear("17 19 GCD")
        self.assertEqual(result, 1)
    
    def test_lcm(self):
        """Test least common multiple"""
        result = self.calc.evaluate_and_clear("12 18 LCM")
        self.assertEqual(result, 36)
    
    def test_frac_exact(self):
        """Test exact fraction conversion"""
        result = self.calc.evaluate_and_clear("0.75 FRAC")
        self.assertEqual(result, 0.75)  # Should convert to 3/4 = 0.75
    
    def test_frac_pi_approximation(self):
        """Test fraction approximation of pi"""
        import math
        result = self.calc.evaluate_and_clear(f"{math.pi} FRAC")
        # Should approximate pi as 355/113 ≈ 3.14159292
        self.assertAlmostEqual(result, 355/113, places=5)


if __name__ == '__main__':
    unittest.main()

