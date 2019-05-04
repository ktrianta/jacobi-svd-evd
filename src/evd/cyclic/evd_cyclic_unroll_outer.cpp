#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include "evd_cyclic.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include "util.hpp"

// I tried to unroll along th rows as well, that got bad perfromance. This was the best, by the diagonal.

void evd_cyclic(struct matrix_t Data_matr, struct matrix_t Data_matr_copy, struct matrix_t Eigen_vectors,
                struct vector_t Eigen_values, int epoch) {
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
                // First unroll
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

            for (i = 0; i < n; ++i) {
                double A_ip = A[n * i + p];
                double A_iq = A[n * i + q];

                A[n * i + p] = c0 * A_ip - s0 * A_iq;
                A[n * i + q] = s0 * A_ip + c0 * A_iq;
            }

            for (i = 0; i < n; ++i) {
                double A_ip0 = A[n * p + i];
                double A_iq0 = A[n * q + i];
                // Working with the transpose of eigenvectors to improve locality.
                double V_pi0 = V[n * p + i];
                double V_qi0 = V[n * q + i];

                A[n * p + i] = c0 * A_ip0 - s0 * A_iq0;
                A[n * q + i] = s0 * A_ip0 + c0 * A_iq0;

                V[n * p + i] = c0 * V_pi0 - s0 * V_qi0;
                V[n * q + i] = s0 * V_pi0 + c0 * V_qi0;
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

int evd_cyclic_tol(struct matrix_t Xmat, struct matrix_t Amat, struct matrix_t Qmat, struct vector_t evec, double tol) {
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
