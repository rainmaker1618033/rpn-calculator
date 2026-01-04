# ============================================================================
# FILE: tests/test_stack_operations.py
# ============================================================================
"""Tests for stack operations"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpn_calculator import Calculator, CalculatorError


class TestStackOperations(unittest.TestCase):
    """Test stack manipulation operations"""
    
    def setUp(self):
        self.calc = Calculator()
    
    def test_swap(self):
        """Test swap operation"""
        self.calc.push(1)
        self.calc.push(2)
        self.calc.operations["SWAP"]()
        self.assertEqual(self.calc.stack, [2, 1])
    
    def test_roll_down(self):
        """Test roll down operation"""
        self.calc.push(1)
        self.calc.push(2)
        self.calc.push(3)
        self.calc.operations["RD"]()
        self.assertEqual(self.calc.stack, [3, 1, 2])
    
    def test_roll_up(self):
        """Test roll up operation"""
        self.calc.push(1)
        self.calc.push(2)
        self.calc.push(3)
        self.calc.operations["RU"]()
        self.assertEqual(self.calc.stack, [2, 3, 1])
    
    def test_clear(self):
        """Test clear operation"""
        self.calc.push(1)
        self.calc.push(2)
        self.calc.push(3)
        self.calc.operations["C"]()
        self.assertEqual(self.calc.stack, [])
    
    def test_delete(self):
        """Test delete operation"""
        self.calc.push(1)
        self.calc.push(2)
        self.calc.operations["DEL"]()
        self.assertEqual(self.calc.stack, [1])
    
    def test_undo(self):
        """Test undo operation"""
        self.calc.evaluate("5 3 +")
        self.calc.operations["UNDO"]()
        self.assertEqual(self.calc.stack, [5, 3])
    
    def test_swap_insufficient_items(self):
        """Test swap with insufficient items"""
        self.calc.push(1)
        with self.assertRaises(CalculatorError):
            self.calc.operations["SWAP"]()


if __name__ == '__main__':
    unittest.main()

