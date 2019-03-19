#pragma once

/**
 * Compute eigenvalue decomposition of a given symmetric, n times n matrix
 * using Jacobi rotations.
 *
 * The algorithm is described in
 * http://web.mit.edu/ehliu/Public/sclark/Golub%20G.H.,%20Van%20Loan%20C.F.-%20Matrix%20Computations.pdf
 * as Algorithm 8.4.2.
 *
 * @param A Matrix of size n times n, given in row-major order, whose EVD will
 * be computed.
 * @param[out] e Output array of size n. Eigenvalues of X will be written to
 * array e,
 * in order of ascending magnitude.
 * @param epoch Number of Jacobi iterations until convergence (default 100)
 */
void evd_classic(struct matrix_t A, struct vector_t e, int epoch = 100);

/**
 * Compute eigenvalue decomposition of a given symmetric, n times n matrix
 * using cyclic Jacobi rotations.
 *
 * The algorithm is described in
 * http://web.mit.edu/ehliu/Public/sclark/Golub%20G.H.,%20Van%20Loan%20C.F.-%20Matrix%20Computations.pdf
 * as Algorithm 8.4.3.
 *
 * @param X Symmetric matrix of size n times n, given in row-major order,
 * whose EVD will be computed.
 * @param n Number of rows/columns of input matrix X.
 * @param e Output array of size n. Eigenvalues of X will be written to array e,
 * in order of ascending magnitude.
 * @param[out] Q Orthonormal matrix of size n times n, outputted in row-major
 * order.  Columns of U form an orthonormal eigenbasis for the column space of
 * X.
 */
void evd_cyclic(const double* const X, int n, double* e, double* Q);