#pragma once
#include "types.hpp"

/**
 * Compute SVD of a given m times n matrix using Jacobi rotations.
 *
 * The algorithm is described in
 * http://www.cs.utexas.edu/users/inderjit/public_papers/HLA_SVD.pdf as
 * Algorithm 7.
 *
 * @param X Matrix of size m times n, given in row-major order, whose SVD will
 * be computed.
 * @param s[out] Output vector of size min(m, n). Singular values of X will be
 * written to array s, in order of descending magnitude.
 * @param[out] U Orthonormal matrix of size m times n, outputted in row-major
 * order. Columns of U are the left singular vectors. ith column of U correspond
 * to the ith singular value in s.
 * @param[out] V Orthonormal matrix of size n times n, outputted in row-major
 * order. Columns of V are the right singular vectors. ith column of V
 * correspond to the ith singular value in s.
 * @param n_iter Number of diagonalization iterations to perform.
 */
void svd(struct matrix_t X, struct vector_t s, struct matrix_t U, struct matrix_t V, size_t n_iter = 1000);
