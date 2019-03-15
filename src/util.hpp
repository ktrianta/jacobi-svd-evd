#pragma once
#include <float.h>

/**
 * Compare two double precision floating point values and return true if they
 * are close in the relative scale.
 *
 * Formally, the function does true if |x - y| < eps * |x + y| is satisfied.
 *
 * @param x Double value.
 * @param y Double value.
 * @param eps Epsilon value to check for closeness.
 */
bool isclose(double x, double y, double eps = DBL_EPSILON);

/**
 * Given three entries of a symmetric matrix X, compute a cosine-sine pair (c, s)
 * such that if B = J(i,j,t)^T*A*J(i,j,t), then B_{ij} = B_{ji} = 0. Here,
 * J(i, j, t) represent the Jacobi rotation matrix constructed using (c, s).
 *
 * The algorithm is described in
 * http://web.mit.edu/ehliu/Public/sclark/Golub%20G.H.,%20Van%20Loan%20C.F.-%20Matrix%20Computations.pdf
 * as Algorithm 8.4.1.
 *
 * @param x_ii Top left entry of the block of X, i.e. X[i][i].
 * @param x_ij Cross entry of the block of X, i.e. X[i][j].
 * @param x_jj Bottom right entry of the block of X, i.e. X[j][j].
 * @param[out] c Cosine value.
 * @param[out] s Sine value.
 */
void sym_jacobi_coeffs(double x_ii, double x_ij, double x_jj, double* c, double* s);
