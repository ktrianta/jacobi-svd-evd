#include "svd.hpp"
#include <assert.h>
#include <math.h>
#include <algorithm>
#include "matrix.hpp"
#include "nsvd.hpp"
#include "types.hpp"
#include "util.hpp"

static void matrix_frobenius(matrix_t m, double* norm, double* off_norm);
static void matrix_off_frobenius(matrix_t m, double* off_norm);

size_t svd(struct matrix_t Amat, struct matrix_t Bmat, struct matrix_t Umat, struct matrix_t Vmat) {
    assert(Amat.rows == Amat.cols);  // Matrix A should be square
    assert(Amat.rows == Bmat.rows && Amat.cols == Bmat.cols);
    assert(Amat.rows == Umat.rows && Amat.cols == Umat.cols);
    assert(Amat.rows == Vmat.rows && Amat.cols == Vmat.cols);

    size_t iter = 0;           // count main loop iterations performed till convergence
    const double tol = 1e-15;  // convergence tolerance
    const size_t n = Amat.rows;
    double* B = Bmat.ptr;
    double* U = Umat.ptr;
    double* V = Vmat.ptr;
    double norm = 0.0;      // frobenius norm of matrix B
    double off_norm = 0.0;  // frobenius norm of the off-diagonal elements of matrix B

    matrix_copy(Bmat, Amat);
    matrix_identity(Umat);
    matrix_identity(Vmat);
    matrix_frobenius(Bmat, &norm, &off_norm);

    // Repeat while the frobenius norm of the off-diagonal elements of matrix B, which is updated in every
    // iteration, is smaller than the forbenius norm of the original matrix B (or A) times the tolerance
    while (off_norm >= tol * norm) {
        for (size_t i = 0; i < n - 1; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const double bii = B[n * i + i];  // B[i][i]
                const double bij = B[n * i + j];  // B[i][j]
                const double bji = B[n * j + i];  // B[j][i]
                const double bjj = B[n * j + j];  // B[j][j]

                // Compute the 2x2 svd of B[i][i], B[i][j], B[j][i] and B[j][j]
                struct svd_2x2_params cf = nsvd(bii, bij, bji, bjj);

                // R_ij(c,s) * B where R_ij(c,s) is the Givens rotation matrix that acts on
                // rows i and j during left multiplication with B
                for (size_t k = 0; k < n; ++k) {
                    double left = cf.c1 * B[n * i + k] - cf.s1 * B[n * j + k];
                    double right = cf.s1 * cf.k * B[n * i + k] + cf.c1 * cf.k * B[n * j + k];
                    B[n * i + k] = left;
                    B[n * j + k] = right;
                }

                // B * R_ij(c',-s') where R_ij(c',-s') is the Givens rotation matrix that acts on
                // columns i and j during right multiplication with B
                for (size_t k = 0; k < n; ++k) {
                    double left = cf.c2 * B[n * k + i] - cf.s2 * B[n * k + j];
                    double right = cf.s2 * B[n * k + i] + cf.c2 * B[n * k + j];
                    B[n * k + i] = left;
                    B[n * k + j] = right;
                }

                // U * R_ij(c,s) where R_ij(c,s) is the Givens rotation matrix that acts on
                // columns i and j during right multiplication with U
                for (size_t k = 0; k < n; ++k) {
                    double left = cf.c1 * U[n * k + i] - cf.s1 * U[n * k + j];
                    double right = cf.s1 * cf.k * U[n * k + i] + cf.c1 * cf.k * U[n * k + j];
                    U[n * k + i] = left;
                    U[n * k + j] = right;
                }

                // V * R_ij(c',-s') where R_ij(c',-s') is the Givens rotation matrix that acts on
                // columns i and j during right multiplication with V
                for (size_t k = 0; k < n; ++k) {
                    double left = cf.c2 * V[n * k + i] - cf.s2 * V[n * k + j];
                    double right = cf.s2 * V[n * k + i] + cf.c2 * V[n * k + j];
                    V[n * k + i] = left;
                    V[n * k + j] = right;
                }
            }
        }

        matrix_off_frobenius(Bmat, &off_norm);
        iter += 1;
    }

    return iter;
}

static void matrix_frobenius(matrix_t m, double* norm, double* off_norm) {
    const size_t M = m.rows;
    const size_t N = m.cols;
    double* data = m.ptr;

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            *norm += data[N * i + j] * data[N * i + j];

            if (i == j) {
                continue;
            } else {
                *off_norm += data[N * i + j] * data[N * i + j];
            }
        }
    }

    *norm = sqrt(*norm);
    *off_norm = sqrt(*off_norm);
}

static void matrix_off_frobenius(matrix_t m, double* off_norm) {
    const size_t M = m.rows;
    const size_t N = m.cols;
    double* data = m.ptr;

    *off_norm = 0.0;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i == j) {
                continue;
            } else {
                *off_norm += data[N * i + j] * data[N * i + j];
            }
        }
    }
    *off_norm = sqrt(*off_norm);
}
