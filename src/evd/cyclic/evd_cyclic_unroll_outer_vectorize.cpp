#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include "evd_cyclic.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include "util.hpp"

void evd_cyclic_unroll_outer_vectorize(struct matrix_t Data_matr, struct matrix_t Data_matr_copy,
                                       struct matrix_t Eigen_vectors, struct vector_t Eigen_values, int epoch) {
    assert(Data_matr.rows == Data_matr.cols);
    double* A = Data_matr_copy.ptr;
    double* V = Eigen_vectors.ptr;
    double* E = Eigen_values.ptr;
    size_t n = Data_matr.rows;
    double c0, s0;
    double c1, s1;

    matrix_identity(Eigen_vectors);
    matrix_copy(Data_matr_copy, Data_matr);

    for (int ep = 0; ep < epoch; ++ep) {
        for (size_t p = 0; p < n - 1; p += 2) {
            for (size_t q = p + 1; q < n - 1; ++q) {
                // Compute cos_t and sin_t for the rotation
                size_t p1 = p + 1;
                size_t q1 = q + 1;
                sym_jacobi_coeffs(A[p * n + p], A[p * n + q], A[q * n + q], &c0, &s0);
                sym_jacobi_coeffs(A[p1 * n + p1], A[p1 * n + q1], A[q1 * n + q1], &c1, &s1);

                size_t i;

                double A_pp = A[n * p + p];
                double A_pq = A[n * q + p];

                A[n * p + p] = c0 * A_pp - s0 * A_pq;
                A[n * q + p] = s0 * A_pp + c0 * A_pq;

                double A_qp = A[n * p + q];
                double A_qq = A[n * q + q];

                A[n * p + q] = c0 * A_qp - s0 * A_qq;
                A[n * q + q] = s0 * A_qp + c0 * A_qq;

                __m256d s_vec = _mm256_set1_pd(s0);
                __m256d c_vec = _mm256_set1_pd(c0);
                __m256d s1_vec = _mm256_set1_pd(s1);
                __m256d c1_vec = _mm256_set1_pd(c1);

                for (i = 0; i < n - 3; i += 4) {
                    __m256d A_piv =
                        _mm256_set_pd(A[n * (i + 3) + p], A[n * (i + 2) + p], A[n * (i + 1) + p], A[n * i + p]);
                    __m256d A_qiv =
                        _mm256_set_pd(A[n * (i + 3) + q], A[n * (i + 2) + q], A[n * (i + 1) + q], A[n * i + q]);
                    __m256d V_piv = _mm256_loadu_pd(V + n * p + i);
                    __m256d V_qiv = _mm256_loadu_pd(V + n * q + i);

                    __m256d sAq_v = _mm256_mul_pd(s_vec, A_qiv);
                    __m256d sAp_v = _mm256_mul_pd(s_vec, A_piv);
                    __m256d cAq_v = _mm256_mul_pd(c_vec, A_qiv);
                    __m256d cAp_v = _mm256_mul_pd(c_vec, A_piv);

                    __m256d nA_piv = _mm256_sub_pd(cAp_v, sAq_v);
                    __m256d nA_qiv = _mm256_add_pd(sAp_v, cAq_v);

                    __m256d sVq_v = _mm256_mul_pd(s_vec, V_qiv);
                    __m256d sVp_v = _mm256_mul_pd(s_vec, V_piv);
                    __m256d cVq_v = _mm256_mul_pd(c_vec, V_qiv);
                    __m256d cVp_v = _mm256_mul_pd(c_vec, V_piv);

                    __m256d nV_piv = _mm256_sub_pd(cVp_v, sVq_v);
                    V_qiv = _mm256_add_pd(sVp_v, cVq_v);

                    A[n * i + p] = nA_piv[0];
                    A[n * (i + 1) + p] = nA_piv[1];
                    A[n * (i + 2) + p] = nA_piv[2];
                    A[n * (i + 3) + p] = nA_piv[3];

                    A[n * i + q] = nA_qiv[0];
                    A[n * (i + 1) + q] = nA_qiv[1];
                    A[n * (i + 2) + q] = nA_qiv[2];
                    A[n * (i + 3) + q] = nA_qiv[3];

                    // since we use load instead of set the indices are reversed.
                    _mm256_storeu_pd(V + n * p + i, nV_piv);
                    _mm256_storeu_pd(V + n * q + i, V_qiv);

                    if (!(i + 3 >= p && i <= p) && !(i + 3 >= q && i <= q)) {
                        _mm256_storeu_pd(A + n * p + i, nA_piv);
                        _mm256_storeu_pd(A + n * q + i, nA_qiv);
                    } else {
                        for (int j = 0; j < 4; j++) {
                            if ((i + j) != p && (i + j) != q) {
                                A[n * p + i + j] = A[n * (i + j) + p];
                                A[n * q + i + j] = A[n * (i + j) + q];
                            }
                        }
                    }
                }

                for (; i < n; ++i) {
                    double V_pi0 = V[n * p + i];
                    double V_qi0 = V[n * q + i];
                    double A_ip0 = A[n * i + p];
                    double A_iq0 = A[n * i + q];

                    double nA_ip = c0 * A_ip0 - s0 * A_iq0;
                    double nA_iq = s0 * A_ip0 + c0 * A_iq0;

                    A[n * i + p] = nA_ip;
                    A[n * i + q] = nA_iq;
                    V[n * p + i] = c0 * V_pi0 - s0 * V_qi0;
                    V[n * q + i] = s0 * V_pi0 + c0 * V_qi0;

                    if (i != p && i != q) {
                        A[n * p + i] = nA_ip;
                        A[n * q + i] = nA_iq;
                    }
                }
                // Second unroll
                A_pp = A[n * p1 + p1];
                A_pq = A[n * q1 + p1];

                A[n * p1 + p1] = c1 * A_pp - s1 * A_pq;
                A[n * q1 + p1] = s1 * A_pp + c1 * A_pq;

                A_qp = A[n * p1 + q1];
                A_qq = A[n * q1 + q1];

                A[n * p1 + q1] = c1 * A_qp - s1 * A_qq;
                A[n * q1 + q1] = s1 * A_qp + c1 * A_qq;

                for (i = 0; i < n - 3; i += 4) {
                    __m256d A_piv =
                        _mm256_set_pd(A[n * (i + 3) + p1], A[n * (i + 2) + p1], A[n * (i + 1) + p1], A[n * i + p1]);
                    __m256d A_qiv =
                        _mm256_set_pd(A[n * (i + 3) + q1], A[n * (i + 2) + q1], A[n * (i + 1) + q1], A[n * i + q1]);

                    __m256d V_piv = _mm256_loadu_pd(V + n * p1 + i);
                    __m256d V_qiv = _mm256_loadu_pd(V + n * q1 + i);

                    __m256d sAq_v = _mm256_mul_pd(s1_vec, A_qiv);
                    __m256d sAp_v = _mm256_mul_pd(s1_vec, A_piv);
                    __m256d cAq_v = _mm256_mul_pd(c1_vec, A_qiv);
                    __m256d cAp_v = _mm256_mul_pd(c1_vec, A_piv);

                    __m256d nA_piv = _mm256_sub_pd(cAp_v, sAq_v);
                    __m256d nA_qiv = _mm256_add_pd(sAp_v, cAq_v);

                    __m256d sVq_v = _mm256_mul_pd(s1_vec, V_qiv);
                    __m256d sVp_v = _mm256_mul_pd(s1_vec, V_piv);
                    __m256d cVq_v = _mm256_mul_pd(c1_vec, V_qiv);
                    __m256d cVp_v = _mm256_mul_pd(c1_vec, V_piv);

                    __m256d nV_piv = _mm256_sub_pd(cVp_v, sVq_v);
                    V_qiv = _mm256_add_pd(sVp_v, cVq_v);

                    A[n * i + p1] = nA_piv[0];
                    A[n * (i + 1) + p1] = nA_piv[1];
                    A[n * (i + 2) + p1] = nA_piv[2];
                    A[n * (i + 3) + p1] = nA_piv[3];

                    A[n * i + q1] = nA_qiv[0];
                    A[n * (i + 1) + q1] = nA_qiv[1];
                    A[n * (i + 2) + q1] = nA_qiv[2];
                    A[n * (i + 3) + q1] = nA_qiv[3];

                    _mm256_storeu_pd(V + n * p1 + i, nV_piv);
                    _mm256_storeu_pd(V + n * q1 + i, V_qiv);

                    if (!(i + 3 >= p1 && i <= p1) && !(i + 3 >= q1 && i <= q1)) {
                        _mm256_storeu_pd(A + n * p1 + i, nA_piv);
                        _mm256_storeu_pd(A + n * q1 + i, nA_qiv);
                    } else {
                        for (int j = 0; j < 4; j++) {
                            if ((i + j) != p1 && (i + j) != q1) {
                                A[n * p1 + i + j] = A[n * (i + j) + p1];
                                A[n * q1 + i + j] = A[n * (i + j) + q1];
                            }
                        }
                    }
                }
                for (; i < n; ++i) {
                    double V_pi0 = V[n * p1 + i];
                    double V_qi0 = V[n * q1 + i];
                    double A_ip0 = A[n * i + p1];
                    double A_iq0 = A[n * i + q1];

                    double nA_ip = c1 * A_ip0 - s1 * A_iq0;
                    double nA_iq = s1 * A_ip0 + c1 * A_iq0;

                    A[n * i + p1] = nA_ip;
                    A[n * i + q1] = nA_iq;
                    V[n * p1 + i] = c1 * V_pi0 - s1 * V_qi0;
                    V[n * q1 + i] = s1 * V_pi0 + c1 * V_qi0;

                    if (i != p1 && i != q1) {
                        A[n * p1 + i] = nA_ip;
                        A[n * q1 + i] = nA_iq;
                    }
                }
            }
        }
        for (size_t p = 0; p < n - 1; p += 2) {
            // Compute cos_t and sin_t for the rotation
            size_t q = n - 1;
            sym_jacobi_coeffs(A[p * n + p], A[p * n + q], A[q * n + q], &c0, &s0);

            size_t i;

            double A_pp = A[n * p + p];
            double A_pq = A[n * q + p];

            A[n * p + p] = c0 * A_pp - s0 * A_pq;
            A[n * q + p] = s0 * A_pp + c0 * A_pq;

            double A_qp = A[n * p + q];
            double A_qq = A[n * q + q];

            A[n * p + q] = c0 * A_qp - s0 * A_qq;
            A[n * q + q] = s0 * A_qp + c0 * A_qq;

            for (i = 0; i < n; ++i) {
                double V_pi0 = V[n * p + i];
                double V_qi0 = V[n * q + i];
                double A_ip0 = A[n * i + p];
                double A_iq0 = A[n * i + q];

                double nA_ip = c0 * A_ip0 - s0 * A_iq0;
                double nA_iq = s0 * A_ip0 + c0 * A_iq0;

                A[n * i + p] = nA_ip;
                A[n * i + q] = nA_iq;
                V[n * p + i] = c0 * V_pi0 - s0 * V_qi0;
                V[n * q + i] = s0 * V_pi0 + c0 * V_qi0;

                if (i != p && i != q) {
                    A[n * p + i] = nA_ip;
                    A[n * q + i] = nA_iq;
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
}
