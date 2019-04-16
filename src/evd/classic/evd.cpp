#include "evd.hpp"
#include <math.h>
#include <stdlib.h>
#include <cassert>
#include "matrix.hpp"
#include "types.hpp"
#include "util.hpp"

void evd_classic(struct matrix_t Data_matr, struct matrix_t Eigen_vectors, struct vector_t Eigen_values, int epoch) {
    assert(Data_matr.rows == Data_matr.cols);

    double* A = Data_matr.ptr;
    double* V = Eigen_vectors.ptr;
    double* E = Eigen_values.ptr;
    const size_t m = Data_matr.rows;

    matrix_identity(Eigen_vectors);

    for (int ep = 1; ep <= epoch; ep++) {
        double val = 0.0;
        int is_not_diagonal = 0;
        int i_max, j_max;
        double alpha, beta, cos_t, sin_t;

        // Find the larget non-diagonal element in the
        // upper triangular matrix

        for (size_t i = 0; i < m; i++) {
            for (size_t j = i + 1; j < m; j++) {
                if (fabs(A[i * m + j]) > val) {
                    i_max = i;
                    j_max = j;
                    val = A[i * m + j];
                    is_not_diagonal = 1;
                }
            }
        }

        if (!is_not_diagonal) break;

        // Compute cos_t and sin_t for the rotation

        alpha = 2.0 * sign(A[i_max * m + i_max] - A[j_max * m + j_max]) * A[i_max * m + j_max];
        beta = fabs(A[i_max * m + i_max] - A[j_max * m + j_max]);
        cos_t = sqrt(0.5 * (1 + beta / sqrt(alpha * alpha + beta * beta)));
        // sin_t = (1 / 2*cos_t) * (alpha / sqrt(alpha*alpha + beta*beta));
        sin_t = sign(alpha) * sqrt(1 - cos_t * cos_t);

        // Corresponding to Jacobi iteration i :

        for (size_t i = 0; i < m; i++) {

            // Compute the eigen values by updating the rows and columns
            // corresponding to the largest off-diagonal entry until convergence

            double A_i_imax = A[m * i + i_max], A_imax_i = A[m * i_max + i];

            A[m * i + i_max] = cos_t * A[m * i + i_max] - sin_t * A[m * i + j_max];
            A[m * i + j_max] = sin_t * A_i_imax + cos_t * A[m * i + j_max];

            A[m * i_max + i] = cos_t * A[m * i_max + i] - sin_t * A[m * j_max + i];
            A[m * j_max + i] = sin_t * A_imax_i + cos_t * A[m * j_max + i];

            // Compute the eigen vectors similarly by updating the eigen vector matrix

            double V_i_imax = V[m * i + i_max];

            V[m * i + i_max] = cos_t * V[m * i + i_max] - sin_t * V[m * i + j_max];
            V[m * i + j_max] = sin_t * V_i_imax + cos_t * V[m * i + j_max];

        }
    }

    // Store the generated eigen values in the vector
    for (size_t i = 0; i < m; i++) {
        E[i] = A[i * m + i];
    }

    reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
}

void evd_classic_tol(struct matrix_t Xmat, struct matrix_t Qmat, struct vector_t evec, double tol) {
    const int n = Xmat.cols;
    double* X = Xmat.ptr;
    double* e = evec.ptr;
    double* Q = Qmat.ptr;
    // A=QtXQ
    double* A = (double*) malloc(sizeof(double) * n * n);

    double offA = 0, eps = 0, abs_a, c, s;
    int p, q;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[n * i + j] = X[n * i + j];
        }
    }
    for (int i = 0; i < n; ++i) {
        Q[n * i + i] = 1.0;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double a_ij = A[n * i + j];
            offA += 2 * a_ij * a_ij;
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            double a_ij = A[n * i + j];
            if (i == j) {
                eps += a_ij * a_ij;
            } else {
                eps += 2 * a_ij * a_ij;
            }
        }
    }
    eps = tol * tol * eps;

    while (offA > eps) {
        abs_a = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double abs_ij = fabs(A[i * n + j]);
                if (abs_ij > abs_a) {
                    abs_a = abs_ij;
                    p = i;
                    q = j;
                }
            }
        }

        sym_jacobi_coeffs(A[p * n + p], A[p * n + q], A[q * n + q], &c, &s);

        double A_ip, A_iq;
        for (int i = 0; i < n; ++i) {
            double Q_ip = Q[n * i + p];
            double Q_iq = Q[n * i + q];
            Q[n * i + p] = c * Q_ip - s * Q_iq;
            Q[n * i + q] = s * Q_ip + c * Q_iq;

            A_ip = A[n * i + p];
            A_iq = A[n * i + q];

            A[n * i + p] = c * A_ip - s * A_iq;
            A[n * i + q] = s * A_ip + c * A_iq;
        }
        for (int i = 0; i < n; ++i) {
            A_ip = A[n * p + i];
            A_iq = A[n * q + i];

            A[n * p + i] = c * A_ip - s * A_iq;
            A[n * q + i] = s * A_ip + c * A_iq;
        }
        offA = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double a_ij = A[n * i + j];
                offA += 2 * a_ij * a_ij;
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        e[i] = A[n * i + i];
    }

    reorder_decomposition(evec, &Qmat, 1, greater);
    free(A);
}
