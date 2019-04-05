#include "evd_cyclic.hpp"
#include <math.h>
#include <stdlib.h>
#include "types.hpp"
#include "util.hpp"

void MatMul(double*, double*, double*, int);

void evd_cyclic(struct matrix_t Data_matr, struct matrix_t Eigen_vectors, struct vector_t Eigen_values, int epoch) {
    double* A = Data_matr.ptr;
    const int m = Data_matr.rows;
    double* V = Eigen_vectors.ptr;
    identity(V, m);

    double* E = Eigen_values.ptr;
    int is_not_diagonal = 0;

    // Build the auxiliary matrices

    double *P, *temp;
    P = (double*) malloc(sizeof(double) * m * m);
    temp = (double*) malloc(sizeof(double) * m * m);

    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < m; j++) {
            if (A[i * m + j] != 0.0) {
                is_not_diagonal = 1;
                break;
            }
        }
    }

    if (is_not_diagonal) {
        for (int ep = 1; ep <= epoch; ep++) {
            double alpha, beta, cos_t, sin_t;
            for (int row = 0; row < m; row++) {
                for (int col = row + 1; col < m; col++) {
                    identity(P, m);

                    // Compute cos_t and sin_t for the rotation matrix

                    alpha = 2.0 * sign(A[row * m + row] - A[col * m + col]) * A[row * m + col];
                    beta = fabs(A[row * m + row] - A[col * m + col]);
                    cos_t = sqrt(0.5 * (1 + beta / sqrt(alpha * alpha + beta * beta)));
                    // sin_t = (1 / 2*cos_t) * (alpha / sqrt(alpha*alpha + beta*beta));
                    sin_t = sign(alpha) * sqrt(1 - cos_t * cos_t);

                    // Initialize the rotation parameters in the identity matrix

                    P[row * m + row] = P[col * m + col] = cos_t;
                    P[col * m + row] = sin_t;
                    P[row * m + col] = -1 * sin_t;

                    // Corresponding to Jacobi iteration i :

                    // 1. Compute the eigen vectors by multiplying with V
                    MatMul(temp, V, P, m);
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < m; j++) {
                            V[i * m + j] = temp[i * m + j];
                        }
                    }

                    // 2. Compute the eigen values by updating A by
                    // performing the operation A(i) = P_t * A(i-1) * P
                    MatMul(temp, A, P, m);
                    transpose(P, P, m);
                    MatMul(A, P, temp, m);
                }
            }
        }
    }

    free(P);
    free(temp);

    // Store the generated eigen values in the vector
    for (int i = 0; i < m; i++) {
        E[i] = A[i * m + i];
    }

    reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
}

void evd_cyclic_tol(struct matrix_t Xmat, struct matrix_t Qmat, struct vector_t evec, double tol) {
    const int n = Xmat.cols;
    double* X = Xmat.ptr;
    double* e = evec.ptr;
    double* Q = Qmat.ptr;
    // A=QtXQ
    double* A = (double*) malloc(sizeof(double) * n * n);

    double offA = 0, eps = 0, c, s;

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
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
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
            }
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

void MatMul(double* P, double* Q, double* R, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                sum += Q[i * n + k] * R[k * n + j];
            }
            P[i * n + j] = sum;
            sum = 0.0;
        }
    }
}
