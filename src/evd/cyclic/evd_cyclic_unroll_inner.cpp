#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include "evd_cyclic.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include "util.hpp"

void evd_cyclic(struct matrix_t Data_matr, struct matrix_t Data_matr_copy, struct matrix_t Eigen_vectors,
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
        for (size_t p = 0; p < n; p++) {
            for (size_t q = p + 1; q < n; q++) {
                // Compute cos_t and sin_t for the rotation
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

                for (i = 0; i < n - 3; i += 4) {
                    double A_ip0 = A[n * i + p];
                    double A_iq0 = A[n * i + q];
                    double A_ip1 = A[n * (i + 1) + p];
                    double A_iq1 = A[n * (i + 1) + q];
                    double A_ip2 = A[n * (i + 2) + p];
                    double A_iq2 = A[n * (i + 2) + q];
                    double A_ip3 = A[n * (i + 3) + p];
                    double A_iq3 = A[n * (i + 3) + q];

                    double nA_ip0 = c0 * A_ip0 - s0 * A_iq0;
                    double nA_iq0 = s0 * A_ip0 + c0 * A_iq0;
                    double nA_ip1 = c0 * A_ip1 - s0 * A_iq1;
                    double nA_iq1 = s0 * A_ip1 + c0 * A_iq1;
                    double nA_ip2 = c0 * A_ip2 - s0 * A_iq2;
                    double nA_iq2 = s0 * A_ip2 + c0 * A_iq2;
                    double nA_ip3 = c0 * A_ip3 - s0 * A_iq3;
                    double nA_iq3 = s0 * A_ip3 + c0 * A_iq3;

                    double V_pi0 = V[n * p + i];
                    double V_qi0 = V[n * q + i];
                    double V_pi1 = V[n * p + i + 1];
                    double V_qi1 = V[n * q + i + 1];
                    double V_pi2 = V[n * p + i + 2];
                    double V_qi2 = V[n * q + i + 2];
                    double V_pi3 = V[n * p + i + 3];
                    double V_qi3 = V[n * q + i + 3];

                    V[n * p + i] = c0 * V_pi0 - s0 * V_qi0;
                    V[n * q + i] = s0 * V_pi0 + c0 * V_qi0;
                    V[n * p + i + 1] = c0 * V_pi1 - s0 * V_qi1;
                    V[n * q + i + 1] = s0 * V_pi1 + c0 * V_qi1;
                    V[n * p + i + 2] = c0 * V_pi2 - s0 * V_qi2;
                    V[n * q + i + 2] = s0 * V_pi2 + c0 * V_qi2;
                    V[n * p + i + 3] = c0 * V_pi3 - s0 * V_qi3;
                    V[n * q + i + 3] = s0 * V_pi3 + c0 * V_qi3;

                    A[n * i + p] = nA_ip0;
                    A[n * i + q] = nA_iq0;
                    A[n * (i + 1) + p] = nA_ip1;
                    A[n * (i + 1) + q] = nA_iq1;
                    A[n * (i + 2) + p] = nA_ip2;
                    A[n * (i + 2) + q] = nA_iq2;
                    A[n * (i + 3) + p] = nA_ip3;
                    A[n * (i + 3) + q] = nA_iq3;

                    if (!(i + 3 >= p && i <= p) && !(i + 3 >= q && i <= q)) {
                        A[n * p + i] = nA_ip0;
                        A[n * q + i] = nA_iq0;
                        A[n * p + i + 1] = nA_ip1;
                        A[n * q + i + 1] = nA_iq1;
                        A[n * p + i + 2] = nA_ip2;
                        A[n * q + i + 2] = nA_iq2;
                        A[n * p + i + 3] = nA_ip3;
                        A[n * q + i + 3] = nA_iq3;
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
