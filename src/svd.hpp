#pragma once

/**
 * Compute SVD of a given m times n matrix using Jacobi rotations.
 *
 * The algorithm is described in
 * http://www.cs.utexas.edu/users/inderjit/public_papers/HLA_SVD.pdf as
 * Algorithm 7.
 *
 * @param X Matrix of size m times n, given in row-major order, whose SVD will
 * be computed.
 * @param m Number of rows of input matrix X.
 * @param n Number of columns of input matrix X.
 * @param s[out] Output array of size min(m, n). Singular values of X will be
 * written to array s, in order of descending magnitude.
 * @param[out] U Orthonormal matrix of size m times n, outputted in row-major
 * order. Columns of U are the left singular vectors.
 * @param[out] V Orthonormal matrix of size n times n, outputted in row-major
 * order. Columns of V are the right singular vectors.
 * @param n_iter Number of diagonalization iterations to perform.
 */
void svd(const double* const X, int m, int n, double* s, double* U, double* V, int n_iter = 1000);
