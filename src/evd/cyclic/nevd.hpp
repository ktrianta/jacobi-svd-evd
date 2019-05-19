#pragma once

/*
 * Hold the coefficients of the normalized Jacobi 2x2 EVD.
 */
struct evd_2x2_params {
    double d1, d2;
    double c1, s1;
    double c2, s2;
    double k;
};

/*
 * Compute the normalized Jacobi EVD of a 2 by 2 matrix.
 *
 * @param w element at position (i,i) of the original matrix.
 * @param x element at position (i,j) of the original matrix.
 * @param y element at position (j,i) of the original matrix.
 * @param z element at position (j,j) of the original matrix.
 * @return coefficients of the 2x2 matrices involved in the
 * 2x2 EVD computation.
 */
struct evd_2x2_params nevd(double w, double x, double y, double z);
