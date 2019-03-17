#include <math.h>
#include <assert.h>
#include "util.hpp"
#include "types.hpp"
#include "svd.hpp"


void svd(struct matrix_t Xmat, struct vector_t svec, struct matrix_t Umat, struct matrix_t Vmat, int n_iter) {
    const int m = Xmat.rows;
    const int n = Xmat.cols;
    const int n_singular_vals = svec.len;

    assert(((m < n) ? m : n) == n_singular_vals);
    assert(m == Umat.rows);
    assert(n == Umat.cols);
    assert(n == Vmat.rows);
    assert(n == Vmat.cols);

    double* X = Xmat.ptr;
    double* s = svec.ptr;
    double* U = Umat.ptr;
    double* V = Vmat.ptr;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            U[n*i + j] = X[n*i + j];
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            V[n*i + j] = 0.0;
            if (i == j) {
                V[n*i + j] = 1.0;
            }
        }
    }

    while (n_iter--) {
        for (int i = 0; i < n-1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double dot_ii = 0, dot_jj = 0, dot_ij = 0;
                for (int k = 0; k < m; ++k) {
                    dot_ii += U[n*k + i] * U[n*k + i];
                    dot_ij += U[n*k + i] * U[n*k + j];
                    dot_jj += U[n*k + j] * U[n*k + j];
                }

                double cosine, sine;
                sym_jacobi_coeffs(dot_ii, dot_ij, dot_jj, &cosine, &sine);
                for (int k = 0; k < m; ++k) {
                    double left = cosine*U[n*k + i] - sine*U[n*k + j];
                    double right = sine*U[n*k + i] + cosine*U[n*k + j];
                    U[n*k + i] = left;
                    U[n*k + j] = right;
                }
                for (int k = 0; k < n; ++k) {
                    double left = cosine*V[n*k + i] - sine*V[n*k + j];
                    double right = sine*V[n*k + i] + cosine*V[n*k + j];
                    V[n*k + i] = left;
                    V[n*k + j] = right;
                }
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        double sigma = 0.0;
        for (int k = 0; k < m; ++k) {
            sigma += U[n*k + i] * U[n*k + i];
        }
        sigma = sqrt(sigma);

        if (i < n_singular_vals) {
            s[i] = sigma;
        }

        for (int k = 0; k < m; ++k) {
            U[n*k + i] /= sigma;
        }
    }

    matrix_t matrices[2] = {Umat, Vmat};
    reorder_decomposition(svec, matrices, 2, greater);
}
