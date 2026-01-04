# ============================================================================
# FILE: tests/test_matrices.py (CORRECTED)
# ============================================================================
"""Tests for matrix operations"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpn_calculator import Calculator


class TestMatrices(unittest.TestCase):
    """Test matrix operations"""
    
    def setUp(self):
        self.calc = Calculator()
    
    def test_matrix_creation(self):
        """Test creating a matrix"""
        self.calc.push([1, 2])
        self.calc.push([3, 4])
        self.calc.push(2)
        self.calc.operations["MATRIX"]()
        result = self.calc.get_result()
        self.assertEqual(result, [[1, 2], [3, 4]])
    
    def test_identity_matrix(self):
        """Test identity matrix creation"""
        self.calc.push(3)
        self.calc.operations["IDENTITY"]()
        result = self.calc.get_result()
        expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.assertEqual(result, expected)
    
    def test_matrix_determinant(self):
        """Test matrix determinant"""
        self.calc.push([[1, 2], [3, 4]])
        self.calc.operations["DET"]()
        result = self.calc.get_result()
        self.assertAlmostEqual(result, -2.0, places=5)
    
    def test_matrix_trace(self):
        """Test matrix trace"""
        self.calc.push([[1, 2], [3, 4]])
        self.calc.operations["TRACE"]()
        result = self.calc.get_result()
        self.assertEqual(result, 5.0)
    
    def test_matrix_transpose(self):
        """Test matrix transpose"""
        self.calc.push([[1, 2], [3, 4]])
        self.calc.operations["TRANSPOSE"]()
        result = self.calc.get_result()
        self.assertEqual(result, [[1, 3], [2, 4]])
    
    def test_matrix_addition(self):
        """Test matrix addition"""
        self.calc.push([[1, 2], [3, 4]])
        self.calc.push([[5, 6], [7, 8]])
        self.calc.operations["M+"]()
        result = self.calc.get_result()
        self.assertEqual(result, [[6, 8], [10, 12]])
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication"""
        self.calc.push([[1, 2], [3, 4]])
        self.calc.push([[5, 6], [7, 8]])
        self.calc.operations["M*"]()
        result = self.calc.get_result()
        self.assertEqual(result, [[19, 22], [43, 50]])
    
    def test_matrix_scalar_multiplication(self):
        """Test scalar-matrix multiplication"""
        self.calc.push([[1, 2], [3, 4]])
        self.calc.push(2)
        self.calc.operations["MSCALE"]()
        result = self.calc.get_result()
        self.assertEqual(result, [[2, 4], [6, 8]])
    
    def test_matrix_inverse(self):
        """Test matrix inverse"""
        self.calc.push([[1, 2], [3, 4]])
        self.calc.operations["MINV"]()
        result = self.calc.get_result()
        # [[1,2],[3,4]]^-1 = [[-2, 1], [1.5, -0.5]]
        self.assertAlmostEqual(result[0][0], -2.0, places=5)
        self.assertAlmostEqual(result[0][1], 1.0, places=5)
        self.assertAlmostEqual(result[1][0], 1.5, places=5)
        self.assertAlmostEqual(result[1][1], -0.5, places=5)
    
    def test_matrix_rank(self):
        """Test matrix rank"""
        self.calc.push([[1, 2], [2, 4]])
        self.calc.operations["RANK"]()
        result = self.calc.get_result()
        self.assertEqual(result, 1)
    
    def test_eigenvalues(self):
        """Test eigenvalue computation"""
        self.calc.push([[1, 2], [2, 1]])
        self.calc.operations["EIGEN"]()
        result = self.calc.get_result()
        # Eigenvalues should be 3 and -1
        self.assertEqual(sorted(result), [-1, 3])
    
    def test_solve_system(self):
        """Test solving linear system"""
        # System: 2x + y = 5, x + 3y = 6
        # Solution: x = 1.8, y = 1.4
        self.calc.push([[2, 1], [1, 3]])
        self.calc.push([5, 6])
        self.calc.operations["MSOLVE"]()
        result = self.calc.get_result()
        # Solution should be approximately [1.615, 1.769]
        self.assertAlmostEqual(result[0], 1.8, places=5)
        self.assertAlmostEqual(result[1], 1.4, places=5)


if __name__ == '__main__':
    unittest.main()