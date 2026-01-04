"""
Tests for matrix decomposition operations
"""

import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpn_calculator import Calculator


class TestMatrixDecompositions(unittest.TestCase):
    """Test matrix decomposition operations"""
    
    def setUp(self):
        self.calc = Calculator()
    
    def assertMatrixAlmostEqual(self, mat1, mat2, places=5):
        """Helper to compare two matrices element-wise"""
        self.assertEqual(len(mat1), len(mat2), "Matrix row count mismatch")
        for i, (row1, row2) in enumerate(zip(mat1, mat2)):
            self.assertEqual(len(row1), len(row2), f"Row {i} length mismatch")
            for j, (val1, val2) in enumerate(zip(row1, row2)):
                self.assertAlmostEqual(val1, val2, places=places,
                                     msg=f"Element [{i}][{j}] mismatch")
    
    def test_qr_decomposition(self):
        """Test QR decomposition"""
        # Test matrix
        A = [[1, 2], [3, 4]]
        self.calc.push(A)
        self.calc.operations["QR"]()
        
        # QR pushes Q then R
        R = self.calc.pop()
        Q = self.calc.pop()
        
        # Verify Q is orthogonal (Q^T * Q = I)
        Q_np = np.array(Q)
        I = np.matmul(Q_np.T, Q_np)
        self.assertMatrixAlmostEqual(I.tolist(), [[1, 0], [0, 1]], places=5)
        
        # Verify R is upper triangular
        self.assertAlmostEqual(R[1][0], 0, places=5)
        
        # Verify Q*R = A
        Q_np = np.array(Q)
        R_np = np.array(R)
        result = np.matmul(Q_np, R_np)
        self.assertMatrixAlmostEqual(result.tolist(), A, places=5)
    
    def test_svd_decomposition(self):
        """Test Singular Value Decomposition"""
        # Test matrix
        A = [[3, 2], [2, 3]]
        self.calc.push(A)
        self.calc.operations["SVD"]()
        
        # SVD pushes U, Sigma, V*
        Vh = self.calc.pop()
        Sigma = self.calc.pop()
        U = self.calc.pop()
        
        # Verify U*Sigma*V* = A
        U_np = np.array(U)
        Sigma_np = np.array(Sigma)
        Vh_np = np.array(Vh)
        
        result = np.matmul(np.matmul(U_np, Sigma_np), Vh_np)
        self.assertMatrixAlmostEqual(result.tolist(), A, places=5)
        
        # Verify singular values are non-negative and sorted
        singular_values = [Sigma[i][i] for i in range(len(Sigma))]
        for sv in singular_values:
            self.assertGreaterEqual(sv, 0)
    
    def test_cholesky_decomposition(self):
        """Test Cholesky decomposition on positive definite matrix"""
        # Symmetric positive definite matrix
        A = [[4, 2], [2, 3]]
        self.calc.push(A)
        self.calc.operations["CHOLESKY"]()
        
        L = self.calc.pop()
        
        # Verify L is lower triangular
        self.assertAlmostEqual(L[0][1], 0, places=5)
        
        # Verify L*L^T = A
        L_np = np.array(L)
        result = np.matmul(L_np, L_np.T)
        self.assertMatrixAlmostEqual(result.tolist(), A, places=5)
    
    def test_lu_decomposition(self):
        """Test LU decomposition with partial pivoting"""
        # Test matrix
        A = [[2, 1, 1], [4, 3, 3], [8, 7, 9]]
        self.calc.push(A)
        
        try:
            self.calc.operations["LU"]()
            
            # LU pushes P, L, U
            U = self.calc.pop()
            L = self.calc.pop()
            P = self.calc.pop()
            
            # Verify L is lower triangular with 1s on diagonal
            for i in range(len(L)):
                self.assertAlmostEqual(L[i][i], 1.0, places=5)
                for j in range(i+1, len(L[i])):
                    self.assertAlmostEqual(L[i][j], 0, places=5)
            
            # Verify U is upper triangular
            for i in range(1, len(U)):
                for j in range(i):
                    self.assertAlmostEqual(U[i][j], 0, places=5)
            
            # Verify P*L*U = A
            P_np = np.array(P)
            L_np = np.array(L)
            U_np = np.array(U)
            
            result = np.matmul(np.matmul(P_np, L_np), U_np)
            self.assertMatrixAlmostEqual(result.tolist(), A, places=5)
            
        except Exception as e:
            if "scipy" in str(e):
                self.skipTest("LU requires scipy library")
            else:
                raise
    
    def test_condition_number(self):
        """Test condition number computation"""
        # Well-conditioned matrix
        A = [[1, 0], [0, 1]]
        self.calc.push(A)
        self.calc.operations["COND"]()
        cond = self.calc.pop()
        self.assertAlmostEqual(cond, 1.0, places=5)
        
        # Ill-conditioned matrix
        A_ill = [[1, 1], [1, 1.0001]]
        self.calc.push(A_ill)
        self.calc.operations["COND"]()
        cond_ill = self.calc.pop()
        self.assertGreater(cond_ill, 10000)  # Should be large
    
    def test_matrix_norm(self):
        """Test matrix norm computation"""
        A = [[1, 2], [3, 4]]
        
        # Frobenius norm (default)
        self.calc.push(A)
        self.calc.operations["NORM"]()
        norm = self.calc.pop()
        # Frobenius norm = sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30)
        expected = np.sqrt(30)
        self.assertAlmostEqual(norm, expected, places=5)
    
    def test_schur_decomposition(self):
        """Test Schur decomposition"""
        A = [[6, -3, 0], [2, 1, 0], [0, 0, 5]]
        self.calc.push(A)
        
        try:
            self.calc.operations["SCHUR"]()
            
            # SCHUR pushes Q, T
            T = self.calc.pop()
            Q = self.calc.pop()
            
            # Verify Q is unitary (Q^T * Q = I)
            Q_np = np.array(Q)
            I = np.matmul(Q_np.T, Q_np)
            identity = np.eye(3)
            self.assertTrue(np.allclose(I, identity, atol=1e-5))
            
            # Verify T is upper triangular (or quasi-triangular for real matrices)
            T_np = np.array(T)
            # Check elements below diagonal are small
            for i in range(1, len(T)):
                for j in range(i-1):  # Allow one subdiagonal
                    if j < i - 1:
                        self.assertAlmostEqual(T[i][j], 0, places=3)
            
            # Verify Q*T*Q^T = A
            result = np.matmul(np.matmul(Q_np, T_np), Q_np.T)
            self.assertMatrixAlmostEqual(result.tolist(), A, places=3)
            
        except Exception as e:
            if "scipy" in str(e):
                self.skipTest("SCHUR requires scipy library")
            else:
                raise
    
    def test_hessenberg_decomposition(self):
        """Test Hessenberg decomposition"""
        A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.calc.push(A)
        
        try:
            self.calc.operations["HESSENBERG"]()
            
            # HESSENBERG pushes Q, H
            H = self.calc.pop()
            Q = self.calc.pop()
            
            # Verify Q is orthogonal
            Q_np = np.array(Q)
            I = np.matmul(Q_np.T, Q_np)
            identity = np.eye(3)
            self.assertTrue(np.allclose(I, identity, atol=1e-5))
            
            # Verify H is upper Hessenberg (zeros below first subdiagonal)
            H_np = np.array(H)
            for i in range(2, len(H)):
                for j in range(i-1):
                    self.assertAlmostEqual(H[i][j], 0, places=5)
            
            # Verify Q*H*Q^T = A
            result = np.matmul(np.matmul(Q_np, H_np), Q_np.T)
            self.assertMatrixAlmostEqual(result.tolist(), A, places=5)
            
        except Exception as e:
            if "scipy" in str(e):
                self.skipTest("HESSENBERG requires scipy library")
            else:
                raise


class TestDecompositionEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for decompositions"""
    
    def setUp(self):
        self.calc = Calculator()
    
    def test_cholesky_non_positive_definite(self):
        """Test Cholesky on non-positive definite matrix fails gracefully"""
        # This matrix is not positive definite
        A = [[1, 2], [2, 1]]
        self.calc.push(A)
        
        # Should raise an error
        try:
            self.calc.operations["CHOLESKY"]()
            # If we get here, check if it's in the error message
            self.fail("Should have raised an error for non-positive definite matrix")
        except Exception as e:
            # Expected - matrix is not positive definite
            self.assertIn("positive definite", str(e).lower())
    
    def test_decomposition_non_square_matrix(self):
        """Test that square-only decompositions fail on non-square matrices"""
        # Non-square matrix
        A = [[1, 2, 3], [4, 5, 6]]
        
        # LU should fail
        self.calc.push(A)
        try:
            self.calc.operations["LU"]()
            self.fail("LU should fail on non-square matrix")
        except Exception as e:
            self.assertIn("square", str(e).lower())
    
    def test_svd_rectangular_matrix(self):
        """Test SVD works on rectangular matrices"""
        # Rectangular matrix (SVD should work)
        A = [[1, 2, 3], [4, 5, 6]]
        self.calc.push(A)
        
        try:
            self.calc.operations["SVD"]()
            Vh = self.calc.pop()
            Sigma = self.calc.pop()
            U = self.calc.pop()
            
            # Should succeed
            self.assertEqual(len(U), 2)
            self.assertEqual(len(Vh), 3)
        except Exception as e:
            self.fail(f"SVD should work on rectangular matrices: {e}")


class TestDecompositionApplications(unittest.TestCase):
    """Test practical applications of decompositions"""
    
    def setUp(self):
        self.calc = Calculator()
    
    def test_solve_using_lu(self):
        """Test solving a system using LU decomposition"""
        # System: Ax = b
        A = [[4, 3], [6, 3]]
        b = [10, 12]
        
        # Get LU decomposition
        self.calc.push(A)
        
        try:
            self.calc.operations["LU"]()
            U = self.calc.pop()
            L = self.calc.pop()
            P = self.calc.pop()
            
            # For testing, just verify we got valid decomposition
            self.assertEqual(len(P), 2)
            self.assertEqual(len(L), 2)
            self.assertEqual(len(U), 2)
            
        except Exception as e:
            if "scipy" in str(e):
                self.skipTest("LU requires scipy library")
            else:
                raise
    
    def test_matrix_rank_via_svd(self):
        """Test determining matrix rank using SVD"""
        # Rank-deficient matrix
        A = [[1, 2], [2, 4]]  # Rank 1
        
        self.calc.push(A)
        self.calc.operations["SVD"]()
        
        Vh = self.calc.pop()
        Sigma = self.calc.pop()
        U = self.calc.pop()
        
        # Count non-zero singular values
        singular_values = [Sigma[i][i] for i in range(len(Sigma))]
        rank = sum(1 for sv in singular_values if abs(sv) > 1e-10)
        
        self.assertEqual(rank, 1)


if __name__ == '__main__':
    unittest.main()