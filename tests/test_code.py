"""
UNIT TESTS FOR MODULAR RPN CALCULATOR
======================================

Directory structure:
tests/
├── __init__.py
├── test_arithmetic.py
├── test_stack_operations.py
├── test_trigonometry.py
├── test_complex_numbers.py
├── test_vectors.py
├── test_matrices.py
├── test_matrix_decompositions.py
├── test_integer_ops.py
├── test_statistics.py
└── run_all_tests.py
"""

# ============================================================================
# FILE: tests/__init__.py
# ============================================================================
"""Test suite for RPN Calculator"""
pass


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


# ============================================================================
# FILE: tests/test_matrices.py
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
        self.calc.evaluate("[1,2] [3,4] 2 MATRIX")
        result = self.calc.get_result()
        self.assertEqual(result, [[1, 2], [3, 4]])
    
    def test_identity_matrix(self):
        """Test identity matrix creation"""
        result = self.calc.evaluate_and_clear("3 IDENTITY")
        expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.assertEqual(result, expected)
    
    def test_matrix_determinant(self):
        """Test matrix determinant"""
        result = self.calc.evaluate_and_clear("[[1,2],[3,4]] DET")
        self.assertAlmostEqual(result, -2.0, places=5)
    
    def test_matrix_trace(self):
        """Test matrix trace"""
        result = self.calc.evaluate_and_clear("[[1,2],[3,4]] TRACE")
        self.assertEqual(result, 5.0)
    
    def test_matrix_transpose(self):
        """Test matrix transpose"""
        result = self.calc.evaluate_and_clear("[[1,2],[3,4]] TRANSPOSE")
        self.assertEqual(result, [[1, 3], [2, 4]])
    
    def test_matrix_addition(self):
        """Test matrix addition"""
        result = self.calc.evaluate_and_clear("[[1,2],[3,4]] [[5,6],[7,8]] M+")
        self.assertEqual(result, [[6, 8], [10, 12]])
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication"""
        result = self.calc.evaluate_and_clear("[[1,2],[3,4]] [[5,6],[7,8]] M*")
        self.assertEqual(result, [[19, 22], [43, 50]])
    
    def test_matrix_scalar_multiplication(self):
        """Test scalar-matrix multiplication"""
        result = self.calc.evaluate_and_clear("[[1,2],[3,4]] 2 MSCALE")
        self.assertEqual(result, [[2, 4], [6, 8]])
    
    def test_matrix_inverse(self):
        """Test matrix inverse"""
        result = self.calc.evaluate_and_clear("[[1,2],[3,4]] MINV")
        # [[1,2],[3,4]]^-1 = [[-2, 1], [1.5, -0.5]]
        self.assertAlmostEqual(result[0][0], -2.0, places=5)
        self.assertAlmostEqual(result[0][1], 1.0, places=5)
        self.assertAlmostEqual(result[1][0], 1.5, places=5)
        self.assertAlmostEqual(result[1][1], -0.5, places=5)
    
    def test_matrix_rank(self):
        """Test matrix rank"""
        result = self.calc.evaluate_and_clear("[[1,2],[2,4]] RANK")
        self.assertEqual(result, 1)
    
    def test_eigenvalues(self):
        """Test eigenvalue computation"""
        result = self.calc.evaluate_and_clear("[[1,2],[2,1]] EIGEN")
        # Eigenvalues should be 3 and -1
        self.assertEqual(sorted(result), [-1, 3])
    
    def test_solve_system(self):
        """Test solving linear system"""
        result = self.calc.evaluate_and_clear("[[2,1],[1,3]] [5,6] MSOLVE")
        # Solution should be approximately [1.615, 1.769]
        self.assertAlmostEqual(result[0], 1.615384615, places=5)
        self.assertAlmostEqual(result[1], 1.769230769, places=5)


if __name__ == '__main__':
    unittest.main()


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


# ============================================================================
# FILE: tests/run_all_tests.py
# ============================================================================
"""Run all tests"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all test modules
from test_arithmetic import TestArithmetic, TestArithmeticErrors
from test_stack_operations import TestStackOperations
from test_complex_numbers import TestComplexNumbers
from test_vectors import TestVectors
from test_matrices import TestMatrices
from test_integer_ops import TestIntegerOps
from test_statistics import TestStatistics
from test_trigonometry import TestTrigonometry


def run_all_tests():
    """Run all test suites"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestArithmetic))
    suite.addTests(loader.loadTestsFromTestCase(TestArithmeticErrors))
    suite.addTests(loader.loadTestsFromTestCase(TestStackOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestComplexNumbers))
    suite.addTests(loader.loadTestsFromTestCase(TestVectors))
    suite.addTests(loader.loadTestsFromTestCase(TestMatrices))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegerOps))
    suite.addTests(loader.loadTestsFromTestCase(TestStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestTrigonometry))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())