#pragma once

/**
 * Compute eigenvalue decomposition of a given symmetric, n times n matrix
 * using Jacobi rotations.
 *
 * The algorithm is described in
 * http://web.mit.edu/ehliu/Public/sclark/Golub%20G.H.,%20Van%20Loan%20C.F.-%20Matrix%20Computations.pdf
 * as Algorithm 8.4.2.
 *
 * @param X Matrix of size n times n, given in row-major order, whose EVD will
 * be computed.
 * @param n Number of rows/columns of input matrix X.
 * @param e Output array of size n. Eigenvalues of X will be written to array e,
 * in order of ascending magnitude.
 * @param[out] Q Orthonormal matrix of size n times n, outputted in row-major
 * order.  Columns of U form an orthonormal eigenbasis for the column space of
 * X.
 */
void evd_classic(const double* const X, int n, double* e, double* Q);

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

/**
 * Given a symmetric, n times n matrix A and integers p, q with 0 <= p < q < n,
 * compute a cosine-sine pair (c, s) such that if B = J(p, q, t)^T A J(p, q, t),
 * then B_{pq} = B_{qp} = 0.
 *
 * The algorithm is described in
 * http://web.mit.edu/ehliu/Public/sclark/Golub%20G.H.,%20Van%20Loan%20C.F.-%20Matrix%20Computations.pdf
 * as Algorithm 8.4.1.
 *
 * @param X Symmetric matrix of size n times n, given in row-major order.
 * @param n Number of rows/columns of input matrix X.
 * @param p First index that satisfies 0 <= p < n.
 * @param q Second index that satisfies p < q < n.
 * @param[out] c Cosine value.
 * @param[out] s Sine value.
 */
void sym_schur2(const double* const X, int n, int p, int q, double* c, double* s);
