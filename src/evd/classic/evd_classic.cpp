#include "evd_classic.hpp"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cassert>
#include "matrix.hpp"
#include "types.hpp"
#include "util.hpp"

void evd_classic(struct matrix_t Data_matr, struct matrix_t Data_matr_copy, struct matrix_t Eigen_vectors,
                 struct vector_t Eigen_values, int epoch) {
    assert(Data_matr.rows == Data_matr.cols);

    // Create a copy of the matrix to prevent modification of the original matrix
    // double* A = (double*) malloc(sizeof(double) * Data_matr.rows * Data_matr.cols);
    double* A = Data_matr_copy.ptr;
    std::copy(Data_matr.ptr, Data_matr.ptr + Data_matr.rows * Data_matr.cols, A);

    double* V = Eigen_vectors.ptr;
    double* E = Eigen_values.ptr;
    const size_t m = Data_matr.rows;

    matrix_identity(Eigen_vectors);

    int is_not_diagonal = 0;

    // Build the auxiliary matrices

    double *P, *temp;
    P = (double*) malloc(sizeof(double) * m * m);
    temp = (double*) malloc(sizeof(double) * m * m);

    for (int ep = 1; ep <= epoch; ep++) {
        double val = 0.0;
        int i_max, j_max;
        double alpha, beta, cos_t, sin_t;

        matrix_identity({P, m, m});

        // Find the larget non-diagonal element in the
        // upper triangular matrix

        for (size_t i = 0; i < m; i++) {
            for (size_t j = i + 1; j < m; j++) {
                alpha = fabs(A[i * m + j]);
                if (alpha > val) {
                    i_max = i;
                    j_max = j;
                    val = alpha;
                    is_not_diagonal = 1;
                }
            }
        }

        if (!is_not_diagonal) break;

        // Compute cos_t and sin_t for the rotation matrix

        alpha = 2.0 * sign(A[i_max * m + i_max] - A[j_max * m + j_max]) * A[i_max * m + j_max];
        beta = fabs(A[i_max * m + i_max] - A[j_max * m + j_max]);
        cos_t = sqrt(0.5 * (1 + beta / sqrt(alpha * alpha + beta * beta)));
        // sin_t = (1 / 2*cos_t) * (alpha / sqrt(alpha*alpha + beta*beta));
        sin_t = sign(alpha) * sqrt(1 - cos_t * cos_t);

        // Initialize the rotation parameters in the identity matrix

        P[i_max * m + i_max] = P[j_max * m + j_max] = cos_t;
        P[j_max * m + i_max] = sin_t;
        P[i_max * m + j_max] = -1 * sin_t;

        // Corresponding to Jacobi iteration i :

        // 1. Compute the eigen vectors by multiplying with V
        matrix_mult({temp, m, m}, {V, m, m}, {P, m, m});
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < m; j++) {
                V[i * m + j] = temp[i * m + j];
            }
        }

        // 2. Compute the eigen values by updating A by
        // performing the operation A(i) = P_t * A(i-1) * P
        matrix_mult({temp, m, m}, {A, m, m}, {P, m, m});
        matrix_transpose({P, m, m}, {P, m, m});
        matrix_mult({A, m, m}, {P, m, m}, {temp, m, m});
    }

    free(P);
    free(temp);

    // Store the generated eigen values in the vector
    for (size_t i = 0; i < m; i++) {
        E[i] = A[i * m + i];
    }

    reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
}

int evd_classic_tol(struct matrix_t Xmat, struct matrix_t Amat, struct matrix_t Qmat, struct vector_t evec,
                     double tol) {
    const int n = Xmat.cols;
    double* e = evec.ptr;
    double* Q = Qmat.ptr;
    double* A = Amat.ptr;
    double offA = 0, eps = 0, c, s;
    int iter = 0;

    matrix_identity(Qmat);
    matrix_copy(Amat, Xmat);
    matrix_frobenius(Amat, &eps, &offA);

    eps = tol * tol * eps;

    while (offA > eps) {
        double abs_a = 0.0;
        int p = 0, q = 1;
        iter++;
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

        matrix_off_frobenius(Amat, &offA);
    }
    for (int i = 0; i < n; ++i) {
        e[i] = A[n * i + i];
    }

    reorder_decomposition(evec, &Qmat, 1, greater);
    return iter;
}
