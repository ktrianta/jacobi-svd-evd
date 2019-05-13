#include "svd_subprocedure.hpp"
#include <math.h>
#include "debug.hpp"
#include "matrix.hpp"
#include "nsvd.hpp"

size_t svd_subprocedure(struct matrix_t Bmat, struct matrix_t Umat, struct matrix_t Vmat) {
    size_t iter = 0;  // count main loop iterations performed till convergence
    size_t n = Bmat.rows;
    const double tol = 1e-15;  // convergence tolerance
    double* B = Bmat.ptr;
    double* U = Umat.ptr;
    double* V = Vmat.ptr;
    double norm = 0.0;      // frobenius norm of matrix B
    double off_norm = 0.0;  // frobenius norm of the off-diagonal elements of matrix B

    matrix_identity(Umat);
    matrix_identity(Vmat);
    matrix_frobenius(Bmat, &norm, &off_norm);

    // Repeat while the frobenius norm of the off-diagonal elements of matrix B, which is updated in every
    // iteration, is smaller than the forbenius norm of the original matrix B (or A) times the tolerance
    while (sqrt(off_norm) > tol * sqrt(norm)) {
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

        matrix_frobenius(Bmat, &norm, &off_norm);
        iter += 1;
    }

    return iter;
}
