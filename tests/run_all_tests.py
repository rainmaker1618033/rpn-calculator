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

# Add this import at the top
from test_matrix_decompositions import (
    TestMatrixDecompositions, 
    TestDecompositionEdgeCases,
    TestDecompositionApplications
)

# Add import
from test_fft_operations import (
    TestFFTOperations,
    TestFFTEdgeCases,
    TestFFTApplications
)

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
    # Add these to the suite
    suite.addTests(loader.loadTestsFromTestCase(TestMatrixDecompositions))
    suite.addTests(loader.loadTestsFromTestCase(TestDecompositionEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestDecompositionApplications))
    # Add to suite
    suite.addTests(loader.loadTestsFromTestCase(TestFFTOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestFFTEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestFFTApplications))
    
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