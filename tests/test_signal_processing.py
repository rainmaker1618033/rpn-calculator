# ============================================================================
# FILE: tests/test_signal_processing.py
# ============================================================================
"""
Test cases for signal processing operations (CONV, CONV2, DECONV, XCORR)
Uses unittest framework (consistent with other tests)
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpn_calculator import Calculator

class TestConvolution(unittest.TestCase):
    """Test 1D convolution operation"""
    
    def setUp(self):
        """Create calculator instance for each test"""
        self.calc = Calculator(enable_logging=False)
    
    def test_simple_convolution(self):
        """Test basic convolution"""
        result = self.calc.evaluate_and_clear("[1,2,3] [1,1,1] CONV")
        self.assertEqual(result, [1, 3, 6, 5, 3])
    
    def test_convolution_commutativity(self):
        """Test that convolution is commutative: a*b = b*a"""
        result1 = self.calc.evaluate_and_clear("[1,2,3] [4,5] CONV")
        result2 = self.calc.evaluate_and_clear("[4,5] [1,2,3] CONV")
        self.assertEqual(result1, result2)
    
    def test_impulse_response(self):
        """Test convolution with impulse"""
        result = self.calc.evaluate_and_clear("[1,0,0,0] [1,2,3] CONV")
        self.assertEqual(result, [1, 2, 3, 0, 0, 0])
    
    def test_moving_average(self):
        """Test moving average filter"""
        result = self.calc.evaluate_and_clear("[1,2,3,4,5] [0.5,0.5] CONV")
        expected = [0.5, 1.5, 2.5, 3.5, 4.5, 2.5]
        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, places=10)
    
    def test_polynomial_multiplication(self):
        """Test polynomial multiplication via convolution"""
        # (x + 1) * (x + 2) = x² + 3x + 2
        # Coefficients: [1,1] * [2,1] = [2, 3, 1]
        result = self.calc.evaluate_and_clear("[1,1] [2,1] CONV")
        self.assertEqual(result, [2, 3, 1])
    
    def test_edge_detection(self):
        """Test edge detection filter"""
        result = self.calc.evaluate_and_clear("[1,1,1,2,2,2] [1,-1] CONV")
        self.assertEqual(result, [1, 0, 0, 1, 0, 0, -2])
    
    def test_single_element(self):
        """Test convolution with single elements"""
        result = self.calc.evaluate_and_clear("[5] [3] CONV")
        self.assertEqual(result, [15])
    
    def test_identity_convolution(self):
        """Test convolution with identity [1]"""
        result = self.calc.evaluate_and_clear("[1,2,3,4] [1] CONV")
        self.assertEqual(result, [1, 2, 3, 4])
    
    def test_zero_padding_effect(self):
        """Test that result length is correct"""
        result = self.calc.evaluate_and_clear("[1,2,3] [1,2] CONV")
        # Length should be 3 + 2 - 1 = 4
        self.assertEqual(len(result), 4)
    
    def test_complex_convolution(self):
        """Test convolution with complex numbers"""
        self.calc.evaluate("[1,1] [1j,1j] CONV")
        result = self.calc.get_result()
        # Should handle complex numbers
        self.assertEqual(len(result), 3)
        self.assertTrue(isinstance(result[0], complex) or result[0] == 1j)
    
    def test_error_empty_vector(self):
        """Test error handling for empty vectors"""
        # Calculator handles errors gracefully without raising exceptions
        self.calc.evaluate("[] [1,2,3] CONV")
        # Stack should remain unchanged after error
        result = self.calc.get_result()
        # Should have the vectors still on stack (error preserved stack)
        self.assertIsNotNone(result)
    
    def test_error_not_enough_operands(self):
        """Test error when stack has insufficient operands"""
        # Calculator handles errors gracefully without raising exceptions
        self.calc.evaluate("[1,2,3] CONV")
        # Stack should still have the vector (error preserved stack)
        result = self.calc.get_result()
        self.assertEqual(result, [1, 2, 3])
    
    def test_error_non_vector(self):
        """Test error when operand is not a vector"""
        # Calculator handles errors gracefully without raising exceptions
        self.calc.evaluate("5 [1,2,3] CONV")
        # Stack should be preserved after error
        self.assertEqual(len(self.calc.stack), 2)  # Both operands still there
    
    def test_convolve_alias(self):
        """Test that CONVOLVE is an alias for CONV"""
        result1 = self.calc.evaluate_and_clear("[1,2] [3,4] CONV")
        result2 = self.calc.evaluate_and_clear("[1,2] [3,4] CONVOLVE")
        self.assertEqual(result1, result2)


class Test2DConvolution(unittest.TestCase):
    """Test 2D convolution operation"""
    
    def setUp(self):
        """Create calculator instance for each test"""
        self.calc = Calculator(enable_logging=False)
    
    def test_simple_2d_conv(self):
        """Test basic 2D convolution"""
        result = self.calc.evaluate_and_clear("[[1,2],[3,4]] [[1,0],[0,1]] CONV2")
        # Should work without errors
        self.assertEqual(len(result), 3)  # Output is 3x3
        self.assertEqual(len(result[0]), 3)
    
    def test_identity_kernel(self):
        """Test 2D convolution with identity kernel"""
        result = self.calc.evaluate_and_clear("[[1,2],[3,4]] [[1]] CONV2")
        # Identity should preserve the image
        self.assertEqual(result, [[1, 2], [3, 4]])
    
    def test_box_filter(self):
        """Test 2x2 box filter (averaging)"""
        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        kernel = [[0.25, 0.25], [0.25, 0.25]]
        # Don't use string formatting - push lists directly
        self.calc.stack.clear()
        self.calc.push(image)
        self.calc.push(kernel)
        self.calc.operations["CONV2"]()
        result = self.calc.get_result()
        # Result should be 4x4
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        self.assertEqual(len(result[0]), 4)
    
    def test_edge_detection_horizontal(self):
        """Test horizontal edge detection (Sobel)"""
        image = [[1, 1, 1], [2, 2, 2], [1, 1, 1]]
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        # Push lists directly
        self.calc.stack.clear()
        self.calc.push(image)
        self.calc.push(sobel_x)
        self.calc.operations["CONV2"]()
        result = self.calc.get_result()
        # Should detect edges
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)  # 3+3-1
        self.assertEqual(len(result[0]), 5)
    
    def test_sharpen_kernel(self):
        """Test sharpening kernel"""
        image = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        sharpen = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
        # Push lists directly
        self.calc.stack.clear()
        self.calc.push(image)
        self.calc.push(sharpen)
        self.calc.operations["CONV2"]()
        result = self.calc.get_result()
        # Should have correct dimensions
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)
        self.assertEqual(len(result[0]), 5)
    
    def test_2d_conv_output_size(self):
        """Test that output size is correct"""
        # 4x4 image, 3x3 kernel → 6x6 output
        image = [[1]*4 for _ in range(4)]
        kernel = [[1]*3 for _ in range(3)]
        # Push lists directly
        self.calc.stack.clear()
        self.calc.push(image)
        self.calc.push(kernel)
        self.calc.operations["CONV2"]()
        result = self.calc.get_result()
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 6)  # 4+3-1
        self.assertEqual(len(result[0]), 6)
    
    def test_single_pixel_image(self):
        """Test convolution with 1x1 image"""
        result = self.calc.evaluate_and_clear("[[5]] [[2]] CONV2")
        self.assertEqual(result, [[10]])
    
    def test_error_empty_matrix(self):
        """Test error handling for empty matrices"""
        # Calculator handles errors gracefully without raising exceptions
        self.calc.evaluate("[[]] [[1,2],[3,4]] CONV2")
        # Stack should be preserved after error
        self.assertIsNotNone(self.calc.get_result())
    
    def test_error_not_matrices(self):
        """Test error when operands are not matrices"""
        # Calculator handles errors gracefully without raising exceptions
        self.calc.evaluate("[1,2,3] [[1,2],[3,4]] CONV2")
        # Stack should be preserved
        self.assertEqual(len(self.calc.stack), 2)
    
    def test_error_not_enough_operands(self):
        """Test error when stack has insufficient operands"""
        # Calculator handles errors gracefully without raising exceptions
        self.calc.evaluate("[[1,2],[3,4]] CONV2")
        # Stack should still have the matrix
        result = self.calc.get_result()
        self.assertEqual(result, [[1, 2], [3, 4]])


class TestDeconvolution(unittest.TestCase):
    """Test deconvolution operation"""
    
    def setUp(self):
        """Create calculator instance for each test"""
        self.calc = Calculator(enable_logging=False)
    
    def test_simple_deconv(self):
        """Test basic deconvolution - recover original signal"""
        # Forward: convolve
        self.calc.evaluate("[1,2,3] [1,1,1] CONV")
        convolved = self.calc.get_result()
        
        # Backward: deconvolve
        result = self.calc.evaluate_and_clear(f"{convolved} [1,1,1] DECONV")
        
        # Should recover [1,2,3]
        self.assertEqual(len(result), 3)
        for r, expected in zip(result, [1, 2, 3]):
            self.assertAlmostEqual(r, expected, places=6)
    
    def test_deconv_identity(self):
        """Test deconvolution with identity"""
        result = self.calc.evaluate_and_clear("[1,2,3,4] [1] DECONV")
        self.assertEqual(result, [1, 2, 3, 4])
    
    def test_polynomial_division(self):
        """Test polynomial division via deconvolution"""
        # (x² + 3x + 2) ÷ (x + 1) = (x + 2)
        # Coefficients: [1, 3, 2] ÷ [1, 1] = [1, 2]
        result = self.calc.evaluate_and_clear("[1,3,2] [1,1] DECONV")
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 1, places=6)
        self.assertAlmostEqual(result[1], 2, places=6)
    
    def test_deconv_roundtrip(self):
        """Test conv → deconv roundtrip"""
        original = [1, 2, 3, 4, 5]
        filter_kernel = [1, 1, 1]
        
        # Forward - push lists directly
        self.calc.stack.clear()
        self.calc.push(original)
        self.calc.push(filter_kernel)
        self.calc.operations["CONV"]()
        convolved = self.calc.get_result()
        self.assertIsNotNone(convolved)
        
        # Backward - push filter kernel
        self.calc.push(filter_kernel)
        self.calc.operations["DECONV"]()
        recovered = self.calc.get_result()
        
        # Should recover original
        self.assertIsNotNone(recovered)
        self.assertEqual(len(recovered), len(original))
        for r, o in zip(recovered, original):
            self.assertAlmostEqual(r, o, places=6)
    
    def test_system_identification(self):
        """Test system identification via deconvolution"""
        # If output = input ⊗ system, then system = output ⊘ input
        input_signal = [1, 0, 0, 0]
        system_response = [1, 2, 3]
        
        # Convolve to get output - push directly
        self.calc.stack.clear()
        self.calc.push(input_signal)
        self.calc.push(system_response)
        self.calc.operations["CONV"]()
        output = self.calc.get_result()
        self.assertIsNotNone(output)
        
        # Recover system response - push input signal
        self.calc.push(input_signal)
        self.calc.operations["DECONV"]()
        recovered_system = self.calc.get_result()
        
        # Should match original system response
        self.assertIsNotNone(recovered_system)
        self.assertEqual(len(recovered_system), len(system_response))
        for r, s in zip(recovered_system, system_response):
            self.assertAlmostEqual(r, s, places=6)
    
    def test_deconv_longer_dividend(self):
        """Test deconvolution with longer dividend"""
        result = self.calc.evaluate_and_clear("[1,3,6,5,3] [1,1,1] DECONV")
        # Should recover [1,2,3]
        self.assertEqual(len(result), 3)
        for r, expected in zip(result, [1, 2, 3]):
            self.assertAlmostEqual(r, expected, places=6)
    
    def test_error_empty_vector(self):
        """Test error handling for empty vectors"""
        # Calculator handles errors gracefully without raising exceptions
        self.calc.evaluate("[] [1,2,3] DECONV")
        # Stack should be preserved after error
        self.assertIsNotNone(self.calc.get_result())
    
    def test_error_not_enough_operands(self):
        """Test error when stack has insufficient operands"""
        # Calculator handles errors gracefully without raising exceptions
        self.calc.evaluate("[1,2,3] DECONV")
        # Stack should still have the vector
        result = self.calc.get_result()
        self.assertEqual(result, [1, 2, 3])
    
    def test_error_non_vector(self):
        """Test error when operand is not a vector"""
        # Calculator handles errors gracefully without raising exceptions
        self.calc.evaluate("5 [1,2,3] DECONV")
        # Stack should be preserved
        self.assertEqual(len(self.calc.stack), 2)


class TestCrossCorrelation(unittest.TestCase):
    """Test cross-correlation operation"""
    
    def setUp(self):
        """Create calculator instance for each test"""
        self.calc = Calculator(enable_logging=False)
    
    def test_simple_xcorr(self):
        """Test basic cross-correlation"""
        result = self.calc.evaluate_and_clear("[1,2,3] [1,1,1] XCORR")
        # Result should be length 3 + 3 - 1 = 5
        self.assertEqual(len(result), 5)
    
    def test_auto_correlation(self):
        """Test auto-correlation (signal with itself)"""
        result = self.calc.evaluate_and_clear("[1,2,3] [1,2,3] XCORR")
        # Peak should be at center
        self.assertEqual(len(result), 5)
        # Center value should be maximum for auto-correlation
        self.assertEqual(result[2], max(result))
    
    def test_xcorr_symmetry(self):
        """Test auto-correlation symmetry"""
        result = self.calc.evaluate_and_clear("[1,2,3,4] [1,2,3,4] XCORR")
        # Auto-correlation should be symmetric
        n = len(result)
        for i in range(n // 2):
            self.assertEqual(result[i], result[n - 1 - i])
    
    def test_impulse_xcorr(self):
        """Test cross-correlation with impulse"""
        result = self.calc.evaluate_and_clear("[1,0,0,0] [1,2,3] XCORR")
        self.assertEqual(len(result), 6)
    
    def test_error_empty_vector(self):
        """Test error handling for empty vectors"""
        # Calculator handles errors gracefully without raising exceptions
        self.calc.evaluate("[] [1,2,3] XCORR")
        # Stack should be preserved after error
        self.assertIsNotNone(self.calc.get_result())
    
    def test_error_not_enough_operands(self):
        """Test error when stack has insufficient operands"""
        # Calculator handles errors gracefully without raising exceptions
        self.calc.evaluate("[1,2,3] XCORR")
        # Stack should still have the vector
        result = self.calc.get_result()
        self.assertEqual(result, [1, 2, 3])


class TestSignalProcessingIntegration(unittest.TestCase):
    """Integration tests combining signal processing with other operations"""
    
    def setUp(self):
        """Create calculator instance for each test"""
        self.calc = Calculator(enable_logging=False)
    
    def test_conv_deconv_chain(self):
        """Test chaining convolution and deconvolution"""
        # Signal → filter → recover
        signal = [1, 2, 3, 4, 5]
        filter_kernel = [1, 1]
        
        # Push signal and filter
        self.calc.stack.clear()
        self.calc.push(signal)
        self.calc.push(filter_kernel)
        self.calc.operations["CONV"]()
        
        # Now deconvolve
        self.calc.push(filter_kernel)
        self.calc.operations["DECONV"]()
        result = self.calc.get_result()
        
        # Should approximately recover original
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(signal))
    
    def test_multiple_convolutions(self):
        """Test chaining multiple convolutions"""
        # Apply two filters in sequence
        self.calc.evaluate("[1,2,3,4,5] [0.5,0.5] CONV")
        self.calc.evaluate("[0.5,0.5] CONV")
        result = self.calc.get_result()
        # Should be twice-smoothed signal
        self.assertGreater(len(result), 0)
    
    def test_conv2_with_vector_ops(self):
        """Test 2D convolution result used in other operations"""
        # Push matrices directly
        self.calc.stack.clear()
        self.calc.push([[1, 2], [3, 4]])
        self.calc.push([[1, 1], [1, 1]])
        self.calc.operations["CONV2"]()
        result = self.calc.get_result()
        # Should be able to get result
        self.assertIsNotNone(result)
        self.assertTrue(self._is_matrix_like(result))
    
    def test_vector_operations_with_conv(self):
        """Test combining vector operations with convolution"""
        # Create a vector, compute convolution, then magnitude
        self.calc.evaluate("[1,2,3] [1,1] CONV")
        self.calc.evaluate("VMAG")
        result = self.calc.get_result()
        # Should get magnitude of convolution result
        self.assertGreater(result, 0)
    
    def test_polynomial_multiply_divide(self):
        """Test polynomial operations using conv and deconv"""
        # Multiply: (x+1)(x+2) = x² + 3x + 2
        self.calc.evaluate("[1,1] [1,2] CONV")
        product = self.calc.get_result()
        self.assertEqual(len(product), 3)
        
        # Divide: result ÷ (x+1) = (x+2)
        self.calc.evaluate("[1,1] DECONV")
        quotient = self.calc.get_result()
        self.assertEqual(len(quotient), 2)
    
    def _is_matrix_like(self, obj):
        """Helper to check if object is matrix-like"""
        return (isinstance(obj, list) and 
                len(obj) > 0 and 
                isinstance(obj[0], list))


# ============================================================================
# MANUAL TEST EXAMPLES (for interactive testing)
# ============================================================================

"""
To manually test the signal processing operations:

1. Start calculator:
   python -m rpn_calculator

2. Try these examples:

   # 1D Convolution
   > [1,2,3] [1,1,1] CONV
   Result: [1, 3, 6, 5, 3]

   # Deconvolution (recover original)
   > [1,3,6,5,3] [1,1,1] DECONV
   Result: [1, 2, 3]

   # 2D Convolution (identity)
   > [[1,2],[3,4]] [[1]] CONV2
   Result: [[1, 2], [3, 4]]

   # 2D Convolution (box blur)
   > [[1,2,3],[4,5,6],[7,8,9]] [[0.25,0.25],[0.25,0.25]] CONV2

   # Edge detection
   > [[1,1,1],[2,2,2],[1,1,1]] [[-1,0,1],[-2,0,2],[-1,0,1]] CONV2

   # Polynomial operations
   > [1,1] [1,2] CONV          # Multiply (x+1)(x+2)
   > [1,1] DECONV              # Divide back

   # Auto-correlation
   > [1,2,3,4] [1,2,3,4] XCORR

3. Check help:
   > HELP SIGNAL
"""


if __name__ == "__main__":
    # Run tests with unittest
    unittest.main()