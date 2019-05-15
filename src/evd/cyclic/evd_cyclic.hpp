#pragma once
#include "types.hpp"

/**
 * Compute eigenvalue decomposition of a given symmetric, n times n matrix
 * using cyclic Jacobi rotations.
 *
 * The algorithm is described in
 * http://web.mit.edu/ehliu/Public/sclark/Golub%20G.H.,%20Van%20Loan%20C.F.-%20Matrix%20Computations.pdf
 * as Algorithm 8.4.3.
 *
 * @param A Matrix of size n times n, given in row-major order, whose EVD will
 * be computed.
 * @param[out] V Matrix of size n times n, given in row-major order, contining
 * the eigenvectors.
 * @param[out] e Output array of size n. Eigenvalues of X will be written to
 * array e,
 * in order of ascending magnitude.
 * @param epoch Number of Jacobi iterations until convergence (default 20)
 */
void evd_cyclic(struct matrix_t Data_matr, struct matrix_t Data_matr_copy, struct matrix_t V, struct vector_t e,
                int epoch = 20);

/**
 * Optimization 1: Vectorized version of the above function
 */
void evd_cyclic_vectorize(struct matrix_t Data_matr, struct matrix_t Data_matr_copy, struct matrix_t V,
                          struct vector_t e, int epoch = 20);

/**
 * Optimization 2: Blocked version of the above function
 */
void evd_cyclic_blocked(struct matrix_t Data_matr, struct matrix_t Data_matr_copy, struct matrix_t V,
                          struct vector_t e, int epoch = 20);
void evd_cyclic_blocked_less_copy(struct matrix_t Data_matr, struct matrix_t Data_matr_copy, struct matrix_t V,
                          struct vector_t e, int epoch = 20);

/**
 * Optimization 3: Blocked version with outer loop unroll of the above function
 */
void evd_cyclic_blocked_unroll_outer(struct matrix_t Data_matr, struct matrix_t Data_matr_copy, struct matrix_t V,
                          struct vector_t e, int epoch = 20);

void evd_cyclic_blocked_unroll_outer_less_copy(struct matrix_t Data_matr, struct matrix_t Data_matr_copy,
                          struct matrix_t V, struct vector_t e, int epoch = 20);

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
int evd_cyclic_tol(struct matrix_t Xmat, struct matrix_t Amat, struct matrix_t Qmat, struct vector_t evec, double tol);

void evd_cyclic_oneloop(struct matrix_t Data_matr, struct matrix_t Data_matr_copy, struct matrix_t V, struct vector_t e,
                        int epoch = 20);

void evd_cyclic_unroll_outer(struct matrix_t Data_matr, struct matrix_t Data_matr_copy, struct matrix_t V,
                             struct vector_t e, int epoch = 20);

void evd_cyclic_unroll_inner(struct matrix_t Data_matr, struct matrix_t Data_matr_copy, struct matrix_t V,
                             struct vector_t e, int epoch = 20);
