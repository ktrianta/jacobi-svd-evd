#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include "evd_cost.hpp"
#include "evd_cyclic.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include "util.hpp"

// I tried to unroll along th rows as well, that got bad perfromance. This was the best, by the diagonal.

size_t evd_cyclic_oneloop_row(struct matrix_t Data_matr, struct matrix_t Data_matr_copy, struct matrix_t Eigen_vectors,
                              struct vector_t Eigen_values, int epoch) {
    assert(Data_matr.rows == Data_matr.cols);
    double* A = Data_matr_copy.ptr;
    double* V = Eigen_vectors.ptr;
    double* E = Eigen_values.ptr;
    size_t n = Data_matr.rows;

    double c0, s0;

    matrix_identity(Eigen_vectors);
    matrix_copy(Data_matr_copy, Data_matr);

    for (int ep = 0; ep < epoch; ++ep) {
        for (size_t p = 0; p < n - 1; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                // Compute cos_t and sin_t for the rotation
                sym_jacobi_coeffs(A[p * n + p], A[p * n + q], A[q * n + q], &c0, &s0);

                size_t i;
                double A_pp = A[n * p + p];
                double A_pq = A[n * q + p];
                double A_qp = A[n * p + q];
                double A_qq = A[n * q + q];

                A[n * p + p] = c0 * A_pp - s0 * A_pq;
                A[n * q + p] = c0 * A_qp - s0 * A_qq;
                A[n * p + q] = s0 * A_pp + c0 * A_pq;
                A[n * q + q] = s0 * A_qp + c0 * A_qq;

                for (i = 0; i < n; ++i) {
                    double V_pi0 = V[n * p + i];
                    double V_qi0 = V[n * q + i];
                    double A_ip0 = A[n * p + i];
                    double A_iq0 = A[n * q + i];

                    double nA_ip = c0 * A_ip0 - s0 * A_iq0;
                    double nA_iq = s0 * A_ip0 + c0 * A_iq0;

                    A[n * p + i] = nA_ip;
                    A[n * q + i] = nA_iq;
                    V[n * p + i] = c0 * V_pi0 - s0 * V_qi0;
                    V[n * q + i] = s0 * V_pi0 + c0 * V_qi0;

                    if (i != p && i != q) {
                        A[n * i + p] = nA_ip;
                        A[n * i + q] = nA_iq;
                    }
                }
            }
        }
    }
    matrix_transpose(Eigen_vectors, Eigen_vectors);
    // Store the generated eigen values in the vector
    for (size_t i = 0; i < n; i++) {
        E[i] = A[i * n + i];
    }
    reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
    return oneloop_cost_evd(n, epoch);
}
