# ============================================================================
# FILE: tests/test_statistics.py
# ============================================================================
"""Tests for statistics operations"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpn_calculator import Calculator


class TestStatistics(unittest.TestCase):
    """Test statistics operations"""
    
    def setUp(self):
        self.calc = Calculator()
    
    def test_combinations(self):
        """Test combinations (n choose k)"""
        result = self.calc.evaluate_and_clear("5 2 COMB")
        self.assertEqual(result, 10)
    
    def test_permutations(self):
        """Test permutations"""
        result = self.calc.evaluate_and_clear("5 2 PERM")
        self.assertEqual(result, 20)
    
    def test_standard_deviation(self):
        """Test standard deviation"""
        result = self.calc.evaluate_and_clear("[1,2,3,4,5] STDV")
        # Std dev of [1,2,3,4,5] is sqrt(2) ≈ 1.414
        self.assertAlmostEqual(result, 1.4142135, places=5)


if __name__ == '__main__':
    unittest.main()

