# ============================================================================
# FILE: tests/test_arithmetic.py
# ============================================================================
"""Tests for arithmetic operations"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpn_calculator import Calculator, CalculatorError


class TestArithmetic(unittest.TestCase):
    """Test basic arithmetic operations"""
    
    def setUp(self):
        """Create a fresh calculator for each test"""
        self.calc = Calculator()
    
    def test_addition(self):
        """Test addition operation"""
        result = self.calc.evaluate_and_clear("3 4 +")
        self.assertEqual(result, 7)
    
    def test_subtraction(self):
        """Test subtraction operation"""
        result = self.calc.evaluate_and_clear("10 3 -")
        self.assertEqual(result, 7)
    
    def test_multiplication(self):
        """Test multiplication operation"""
        result = self.calc.evaluate_and_clear("6 7 *")
        self.assertEqual(result, 42)
    
    def test_division(self):
        """Test division operation"""
        result = self.calc.evaluate_and_clear("15 3 /")
        self.assertEqual(result, 5.0)
    
    def test_division_by_zero(self):
        """Test that division by zero raises error"""
        self.calc.push(5)
        self.calc.push(0)
        with self.assertRaises(CalculatorError):
            self.calc.operations["/"]()
    
    def test_power(self):
        """Test exponentiation"""
        result = self.calc.evaluate_and_clear("2 8 ^")
        self.assertEqual(result, 256)
    
    def test_modulo(self):
        """Test modulo operation"""
        result = self.calc.evaluate_and_clear("17 5 MOD")
        self.assertEqual(result, 2)
    
    def test_parallel(self):
        """Test parallel operation (product over sum)"""
        result = self.calc.evaluate_and_clear("6 3 ||")
        self.assertEqual(result, 2.0)
    
    def test_chained_operations(self):
        """Test multiple operations in sequence"""
        result = self.calc.evaluate_and_clear("5 3 + 2 *")
        self.assertEqual(result, 16)
    
    def test_vector_addition(self):
        """Test element-wise vector addition"""
        result = self.calc.evaluate_and_clear("[1,2,3] [4,5,6] +")
        self.assertEqual(result, [5, 7, 9])
    
    def test_vector_scalar_multiplication(self):
        """Test scalar-vector multiplication"""
        result = self.calc.evaluate_and_clear("[2,3,4] 2 *")
        self.assertEqual(result, [4, 6, 8])


class TestArithmeticErrors(unittest.TestCase):
    """Test error handling in arithmetic"""
    
    def setUp(self):
        self.calc = Calculator()
    
    def test_insufficient_operands_addition(self):
        """Test addition with insufficient operands"""
        self.calc.push(5)
        with self.assertRaises(CalculatorError):
            self.calc.operations["+"]()
    
    def test_insufficient_operands_empty_stack(self):
        """Test operation on empty stack"""
        with self.assertRaises(CalculatorError):
            self.calc.operations["+"]()


if __name__ == '__main__':
    unittest.main()
