# ============================================================================
# FILE: tests/test_vectors.py
# ============================================================================
"""Tests for vector operations"""

import unittest
import sys
import os
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpn_calculator import Calculator


class TestVectors(unittest.TestCase):
    """Test vector operations"""
    
    def setUp(self):
        self.calc = Calculator()
    
    def test_dot_product(self):
        """Test dot product"""
        result = self.calc.evaluate_and_clear("[1,2,3] [4,5,6] DOT")
        self.assertEqual(result, 32)
    
    def test_vector_magnitude(self):
        """Test vector magnitude"""
        result = self.calc.evaluate_and_clear("[3,4] VMAG")
        self.assertEqual(result, 5.0)
    
    def test_cross_product(self):
        """Test 3D cross product"""
        result = self.calc.evaluate_and_clear("[1,0,0] [0,1,0] VCROSS")
        self.assertEqual(result, [0, 0, 1])
    
    def test_cross_product_2(self):
        """Test cross product with different vectors"""
        result = self.calc.evaluate_and_clear("[1,2,3] [4,5,6] VCROSS")
        self.assertEqual(result, [-3, 6, -3])
    
    def test_vector_normalization(self):
        """Test vector normalization"""
        result = self.calc.evaluate_and_clear("[3,4] VNORM")
        self.assertAlmostEqual(result[0], 0.6, places=5)
        self.assertAlmostEqual(result[1], 0.8, places=5)
    
    def test_3d_vector_normalization(self):
        """Test 3D vector normalization"""
        result = self.calc.evaluate_and_clear("[1,2,2] VNORM")
        expected_mag = 3.0
        self.assertAlmostEqual(result[0], 1/expected_mag, places=5)
        self.assertAlmostEqual(result[1], 2/expected_mag, places=5)
        self.assertAlmostEqual(result[2], 2/expected_mag, places=5)


if __name__ == '__main__':
    unittest.main()

