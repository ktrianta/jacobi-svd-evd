#include "evd_cyclic.hpp"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include "matrix.hpp"
#include "types.hpp"
#include "util.hpp"

void evd_cyclic(struct matrix_t Data_matr, struct matrix_t Data_matr_copy, struct matrix_t Eigen_vectors,
                struct vector_t Eigen_values, int epoch) {
    assert(Data_matr.rows == Data_matr.cols);
    double* A = Data_matr_copy.ptr;
    // Create a copy of the matrix to prevent modification of the original matrix
    memcpy(A, Data_matr.ptr, Data_matr.rows * Data_matr.cols * sizeof(double));

    double* V = Eigen_vectors.ptr;
    double* E = Eigen_values.ptr;
    const size_t m = Data_matr.rows;

    matrix_identity(Eigen_vectors);

    int is_not_diagonal = 0;

    for (int ep = 1; ep <= epoch; ep++) {
        double alpha, beta, cos_t, sin_t;

        for (size_t i = 0; i < m; i++) {
            for (size_t j = i + 1; j < m; j++) {
                if (A[i * m + j] != 0.0) {
                    is_not_diagonal = 1;
                    break;
                }
            }
        }

        if (!is_not_diagonal) break;

        for (size_t row = 0; row < m; row++) {
            for (size_t col = row + 1; col < m; col++) {
                // Compute cos_t and sin_t for the rotation

                alpha = 2.0 * sign(A[row * m + row] - A[col * m + col]) * A[row * m + col];
                beta = fabs(A[row * m + row] - A[col * m + col]);
                cos_t = sqrt(0.5 * (1 + beta / sqrt(alpha * alpha + beta * beta)));
                // sin_t = (1 / 2*cos_t) * (alpha / sqrt(alpha*alpha + beta*beta));
                sin_t = sign(alpha) * sqrt(1 - cos_t * cos_t);

                for (size_t i = 0; i < m; i++) {
                    // Compute the eigen values by updating the rows until convergence

                    double A_i_r = A[m * i + row];
                    A[m * i + row] = cos_t * A[m * i + row] + sin_t * A[m * i + col];
                    A[m * i + col] = cos_t * A[m * i + col] - sin_t * A_i_r;
                }

                for (size_t i = 0; i < m; i++) {
                    // Compute the eigen values by updating the columns until convergence

                    double A_r_i = A[m * row + i];
                    A[m * row + i] = cos_t * A[m * row + i] + sin_t * A[m * col + i];
                    A[m * col + i] = cos_t * A[m * col + i] - sin_t * A_r_i;

                    // Compute the eigen vectors similarly by updating the eigen vector matrix
                    double V_i_r = V[m * i + row];
                    V[m * i + row] = cos_t * V[m * i + row] + sin_t * V[m * i + col];
                    V[m * i + col] = cos_t * V[m * i + col] - sin_t * V_i_r;
                }
            }
        }
        is_not_diagonal = 0;
    }

    // Store the generated eigen values in the vector
    for (size_t i = 0; i < m; i++) {
        E[i] = A[i * m + i];
    }

    reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
}

int evd_cyclic_tol(struct matrix_t Xmat, struct matrix_t Amat, struct matrix_t Qmat, struct vector_t evec,
                    double tol) {
    const size_t n = Xmat.cols;
    double* e = evec.ptr;
    double* Q = Qmat.ptr;
    double* A = Amat.ptr;
    double offA = 0, eps = 0, c, s;
    int iter = 0;
    // A=QtXQ

    matrix_identity(Qmat);
    matrix_copy(Amat, Xmat);
    matrix_frobenius(Amat, &eps, &offA);

    eps = tol * tol * eps;

    while (offA > eps) {
        iter++;
        for (size_t p = 0; p < n; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                sym_jacobi_coeffs(A[p * n + p], A[p * n + q], A[q * n + q], &c, &s);

                double A_ip, A_iq;
                for (size_t i = 0; i < n; ++i) {
                    double Q_ip = Q[n * i + p];
                    double Q_iq = Q[n * i + q];
                    Q[n * i + p] = c * Q_ip - s * Q_iq;
                    Q[n * i + q] = s * Q_ip + c * Q_iq;

                    A_ip = A[n * i + p];
                    A_iq = A[n * i + q];

                    A[n * i + p] = c * A_ip - s * A_iq;
                    A[n * i + q] = s * A_ip + c * A_iq;
                }
                for (size_t i = 0; i < n; ++i) {
                    A_ip = A[n * p + i];
                    A_iq = A[n * q + i];

                    A[n * p + i] = c * A_ip - s * A_iq;
                    A[n * q + i] = s * A_ip + c * A_iq;
                }
            }
        }
        matrix_off_frobenius(Amat, &offA);
    }
    for (size_t i = 0; i < n; ++i) {
        e[i] = A[n * i + i];
    }

    reorder_decomposition(evec, &Qmat, 1, greater);
    return iter;
}
