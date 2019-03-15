#include <math.h>
#include "util.hpp"
#include "svd.hpp"


void svd(const double* const X, int m, int n, double* s, double* U, double* V, int n_iter) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            U[n*i + j] = X[n*i + j];
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

        if (i < (m < n ? m : n)) {
            s[i] = sigma;
        }

        for (int k = 0; k < m; ++k) {
            U[n*k + i] /= sigma;
        }
    }
}
