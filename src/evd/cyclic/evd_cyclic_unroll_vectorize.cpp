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

                // Vectorization inputs
                __m256d cos_c1, sin_c1, cos_c2, sin_c2;
                __m256d nV_pi, nV_qi;
                // __m256d nA_ip, nA_iq;
                __m256d sin_vec0, cos_vec0, sin_vec1, cos_vec1;
                sin_vec0 = _mm256_set1_pd(s0);
                cos_vec0 = _mm256_set1_pd(c0);
                sin_vec1 = _mm256_set1_pd(s1);
                cos_vec1 = _mm256_set1_pd(c1);

                size_t i;
                // First unroll
                double A_pp = A[n * p + p];
                double A_pq = A[n * q + p];

                A[n * p + p] = c0 * A_pp - s0 * A_pq;
                A[n * q + p] = s0 * A_pp + c0 * A_pq;

                double A_qp = A[n * p + q];
                double A_qq = A[n * q + q];

                A[n * p + q] = c0 * A_qp - s0 * A_qq;
                A[n * q + q] = s0 * A_qp + c0 * A_qq;

                for (i = 0; i < n; i += 1) {
                    double A_ip0 = A[n * i + p];
                    double A_iq0 = A[n * i + q];

                    double nA_ip = c0 * A_ip0 - s0 * A_iq0;
                    double nA_iq = s0 * A_ip0 + c0 * A_iq0;

                    A[n * i + p] = nA_ip;
                    A[n * i + q] = nA_iq;

                    if (i != p && i != q) {
                        A[n * p + i] = A[n * i + p];
                        A[n * q + i] = A[n * i + q];
                    }

                    /*__m256d A_ip0, A_iq0;
                    A_ip0 = _mm256_set_pd(A[n * i + p], A[n * i + n + p], A[n * i + n * 2 + p],
                                          A[n * i + n * 3 + p]);
                    A_iq0 = _mm256_set_pd(A[n * i + q], A[n * i + n + q], A[n * i + n * 2 + q],
                                          A[n * i + n * 3 + q]);
                    cos_c1 = _mm256_mul_pd(A_ip0, cos_vec0);
                    sin_c1 = _mm256_mul_pd(A_iq0, sin_vec0);
                    nA_ip = _mm256_sub_pd(cos_c1, sin_c1);

                    sin_c2 = _mm256_mul_pd(A_ip0, sin_vec0);
                    cos_c2 = _mm256_mul_pd(A_iq0, cos_vec0);
                    nA_iq = _mm256_add_pd(sin_c2, cos_c2);

                    double* nA_ip_update = (double*) &nA_ip;
                    A[n * i + p] = nA_ip_update[3];
                    A[n * i + n + p] = nA_ip_update[2];
                    A[n * i + n * 2 + p] = nA_ip_update[1];
                    A[n * i + n * 3 + p] = nA_ip_update[0];

                    double* nA_iq_update = (double*) &nA_iq;
                    A[n * i + q] = nA_iq_update[3];
                    A[n * i + n + q] = nA_iq_update[2];
                    A[n * i + n * 2 + q] = nA_iq_update[1];
                    A[n * i + n * 3 + q] = nA_iq_update[0];

                    if (i != p && i != q) {
                        // _mm256_storeu_pd(A + n * p + i, nA_ip);
                        // _mm256_storeu_pd(A + n * q + i, nA_iq);
                        A[n * p + i] = A[n * i + p];
                        A[n * q + i] = A[n * i + q];
                        A[n * p + i + 1] = A[n * (i+1) + p];
                        A[n * q + i + 1] = A[n * (i+1) + q];
                        A[n * p + i + 2] = A[n * (i+2) + p];
                        A[n * q + i + 2] = A[n * (i+2) + q];
                        A[n * p + i + 3] = A[n * (i+3) + p];
                        A[n * q + i + 3] = A[n * (i+3) + q];
                    }*/
                }

                for (i = 0; i < n; i += 4) {
                    __m256d V_pi0, V_qi0;

                    V_pi0 = _mm256_loadu_pd(V + n * p + i);
                    V_qi0 = _mm256_loadu_pd(V + n * q + i);

                    cos_c1 = _mm256_mul_pd(V_pi0, cos_vec0);
                    sin_c1 = _mm256_mul_pd(V_qi0, sin_vec0);
                    nV_pi = _mm256_sub_pd(cos_c1, sin_c1);
                    _mm256_storeu_pd(V + n * p + i, nV_pi);

                    sin_c2 = _mm256_mul_pd(V_pi0, sin_vec0);
                    cos_c2 = _mm256_mul_pd(V_qi0, cos_vec0);
                    nV_qi = _mm256_add_pd(sin_c2, cos_c2);
                    _mm256_storeu_pd(V + n * q + i, nV_qi);
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

                for (i = 0; i < n; ++i) {
                    double A_ip0 = A[n * i + p1];
                    double A_iq0 = A[n * i + q1];

                    double nA_ip = c1 * A_ip0 - s1 * A_iq0;
                    double nA_iq = s1 * A_ip0 + c1 * A_iq0;

                    A[n * i + p1] = nA_ip;
                    A[n * i + q1] = nA_iq;

                    if (i != p1 && i != q1) {
                        A[n * p1 + i] = nA_ip;
                        A[n * q1 + i] = nA_iq;
                    }
                }

                for (i = 0; i < n; i += 4) {
                    __m256d V_pi1, V_qi1;

                    V_pi1 = _mm256_loadu_pd(V + n * p1 + i);
                    V_qi1 = _mm256_loadu_pd(V + n * q1 + i);

                    cos_c1 = _mm256_mul_pd(V_pi1, cos_vec1);
                    sin_c1 = _mm256_mul_pd(V_qi1, sin_vec1);
                    nV_pi = _mm256_sub_pd(cos_c1, sin_c1);
                    _mm256_storeu_pd(V + n * p1 + i, nV_pi);

                    sin_c2 = _mm256_mul_pd(V_pi1, sin_vec1);
                    cos_c2 = _mm256_mul_pd(V_qi1, cos_vec1);
                    nV_qi = _mm256_add_pd(sin_c2, cos_c2);
                    _mm256_storeu_pd(V + n * q1 + i, nV_qi);
                }
            }
        }

        for (size_t p = 0; p < n - 1; p += 2) {
            // Compute cos_t and sin_t for the rotation
            size_t q = n - 1;
            sym_jacobi_coeffs(A[p * n + p], A[p * n + q], A[q * n + q], &c0, &s0);

            __m256d sin_vec0, cos_vec0;
            __m256d cos_c1, sin_c1, cos_c2, sin_c2;
            __m256d nA_ip, nA_iq, nV_ip, nV_iq;
            sin_vec0 = _mm256_set1_pd(s0);
            cos_vec0 = _mm256_set1_pd(c0);

            size_t i;

            for (i = 0; i < n; i += 4) {
                __m256d A_ip, A_iq;
                A_ip = _mm256_set_pd(A[n * i + p], A[n * i + n + p], A[n * i + n * 2 + p], A[n * i + n * 3 + p]);
                A_iq = _mm256_set_pd(A[n * i + q], A[n * i + n + q], A[n * i + n * 2 + q], A[n * i + n * 3 + q]);
                cos_c1 = _mm256_mul_pd(A_ip, cos_vec0);
                sin_c1 = _mm256_mul_pd(A_iq, sin_vec0);
                nA_ip = _mm256_sub_pd(cos_c1, sin_c1);

                sin_c2 = _mm256_mul_pd(A_ip, sin_vec0);
                cos_c2 = _mm256_mul_pd(A_iq, cos_vec0);
                nA_iq = _mm256_add_pd(sin_c2, cos_c2);

                double* nA_ip_update = (double*) &nA_ip;
                A[n * i + p] = nA_ip_update[3];
                A[n * i + n + p] = nA_ip_update[2];
                A[n * i + n * 2 + p] = nA_ip_update[1];
                A[n * i + n * 3 + p] = nA_ip_update[0];

                double* nA_iq_update = (double*) &nA_iq;
                A[n * i + q] = nA_iq_update[3];
                A[n * i + n + q] = nA_iq_update[2];
                A[n * i + n * 2 + q] = nA_iq_update[1];
                A[n * i + n * 3 + q] = nA_iq_update[0];
            }

            for (i = 0; i < n; i += 4) {
                __m256d A_pi, A_qi;
                A_pi = _mm256_loadu_pd(A + n * p + i);
                A_qi = _mm256_loadu_pd(A + n * q + i);

                cos_c1 = _mm256_mul_pd(A_pi, cos_vec0);
                sin_c1 = _mm256_mul_pd(A_qi, sin_vec0);
                nA_ip = _mm256_sub_pd(cos_c1, sin_c1);
                _mm256_storeu_pd(A + n * p + i, nA_ip);

                sin_c2 = _mm256_mul_pd(A_pi, sin_vec0);
                cos_c2 = _mm256_mul_pd(A_qi, cos_vec0);
                nA_iq = _mm256_add_pd(sin_c2, cos_c2);
                _mm256_storeu_pd(A + n * q + i, nA_iq);
            }
            for (i = 0; i < n; i += 4) {
                // Working with the transpose of eigenvectors to improve locality.
                __m256d V_pi, V_qi;
                V_pi = _mm256_loadu_pd(V + n * p + i);
                V_qi = _mm256_loadu_pd(V + n * q + i);

                cos_c1 = _mm256_mul_pd(V_pi, cos_vec0);
                sin_c1 = _mm256_mul_pd(V_qi, sin_vec0);
                nV_ip = _mm256_sub_pd(cos_c1, sin_c1);
                _mm256_storeu_pd(V + n * p + i, nV_ip);

                sin_c2 = _mm256_mul_pd(V_pi, sin_vec0);
                cos_c2 = _mm256_mul_pd(V_qi, cos_vec0);
                nV_iq = _mm256_add_pd(sin_c2, cos_c2);
                _mm256_storeu_pd(V + n * q + i, nV_iq);
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
