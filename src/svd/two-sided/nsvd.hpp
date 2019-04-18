#pragma once

/*
 * Hold the coefficients of the normalized Jacobi 2x2 SVD.
 */
struct svd_2x2_params {
    double d1, d2;
    double c1, s1;
    double c2, s2;
    double k;
};

/*
 * Compute the normalized Jacobi SVD of a 2 by 2 matrix.
 *
 * The algorithm is called NSVD and can be found in
 * https://maths-people.anu.edu.au/~brent/pd/rpb080i.pdf
 *
 * @param w element at position (i,i) of the original matrix.
 * @param x element at position (i,j) of the original matrix.
 * @param y element at position (j,i) of the original matrix.
 * @param z element at position (j,j) of the original matrix.
 * @return coefficients of the 2x2 matrices involved in the 2x2 SVD computation.
 */
struct svd_2x2_params nsvd(double w, double x, double y, double z);
