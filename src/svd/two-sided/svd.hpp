#pragma once

#include "types.hpp"

/**
 * Compute SVD of a given n times n matrix using Jacobi rotations.
 *
 * The algorithm is described as algorithm 7 in
 * http://www.cs.utexas.edu/users/inderjit/public_papers/HLA_SVD.pdf
 * A more detailed and rigorous definition can be found in
 * http://www.netlib.org/utk/people/JackDongarra/PAPERS/svd-sirev-M111773R.pdf
 *
 * @param X      Input matrix of size n times n, given in row-major order.
 * @param[out] B Diagonal matrix of size n times n, in row-major order.
 *               The diagonal consists of the singular values of X in order
 *               of descending magnitude.
 * @param[out] U Orthonormal matrix of size n times n, in row-major order.
 *               Columns of U are the left singular vectors and the ith column
 *               corresponds to the ith singular value in s.
 * @param[out] V Orthonormal matrix of size n times n, in row-major order.
 *               Columns of V are the right singular vectors and the ith column
 *               corresponds to the ith singular value in s.
 * @return       Total number of executed floating point operations.
 */
size_t svd(struct matrix_t X, struct matrix_t B, struct matrix_t U, struct matrix_t V);

size_t svd_blocked(struct matrix_t X, struct matrix_t B, struct matrix_t U, struct matrix_t V);
