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

    while (sqrt(off_norm) > tol * sqrt(norm)) {
        for (size_t i = 0; i < n - 1; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const double bii = B[n * i + i];
                const double bij = B[n * i + j];
                const double bji = B[n * j + i];
                const double bjj = B[n * j + j];
                struct svd_2x2_params cf = nsvd(bii, bij, bji, bjj);

                size_t k = 0;
                for (; k + 7 < n; k += 8) {
                    double b_ik_0 = B[n * i + (k + 0)];
                    double b_jk_0 = B[n * j + (k + 0)];
                    double b_ik_1 = B[n * i + (k + 1)];
                    double b_jk_1 = B[n * j + (k + 1)];
                    double b_ik_2 = B[n * i + (k + 2)];
                    double b_jk_2 = B[n * j + (k + 2)];
                    double b_ik_3 = B[n * i + (k + 3)];
                    double b_jk_3 = B[n * j + (k + 3)];
                    double b_ik_4 = B[n * i + (k + 4)];
                    double b_jk_4 = B[n * j + (k + 4)];
                    double b_ik_5 = B[n * i + (k + 5)];
                    double b_jk_5 = B[n * j + (k + 5)];
                    double b_ik_6 = B[n * i + (k + 6)];
                    double b_jk_6 = B[n * j + (k + 6)];
                    double b_ik_7 = B[n * i + (k + 7)];
                    double b_jk_7 = B[n * j + (k + 7)];

                    double left_0 = cf.c1 * b_ik_0 - cf.s1 * b_jk_0;
                    double right_0 = cf.s1 * cf.k * b_ik_0 + cf.c1 * cf.k * b_jk_0;
                    double left_1 = cf.c1 * b_ik_1 - cf.s1 * b_jk_1;
                    double right_1 = cf.s1 * cf.k * b_ik_1 + cf.c1 * cf.k * b_jk_1;
                    double left_2 = cf.c1 * b_ik_2 - cf.s1 * b_jk_2;
                    double right_2 = cf.s1 * cf.k * b_ik_2 + cf.c1 * cf.k * b_jk_2;
                    double left_3 = cf.c1 * b_ik_3 - cf.s1 * b_jk_3;
                    double right_3 = cf.s1 * cf.k * b_ik_3 + cf.c1 * cf.k * b_jk_3;
                    double left_4 = cf.c1 * b_ik_4 - cf.s1 * b_jk_4;
                    double right_4 = cf.s1 * cf.k * b_ik_4 + cf.c1 * cf.k * b_jk_4;
                    double left_5 = cf.c1 * b_ik_5 - cf.s1 * b_jk_5;
                    double right_5 = cf.s1 * cf.k * b_ik_5 + cf.c1 * cf.k * b_jk_5;
                    double left_6 = cf.c1 * b_ik_6 - cf.s1 * b_jk_6;
                    double right_6 = cf.s1 * cf.k * b_ik_6 + cf.c1 * cf.k * b_jk_6;
                    double left_7 = cf.c1 * b_ik_7 - cf.s1 * b_jk_7;
                    double right_7 = cf.s1 * cf.k * b_ik_7 + cf.c1 * cf.k * b_jk_7;

                    B[n * i + (k + 0)] = left_0;
                    B[n * j + (k + 0)] = right_0;
                    B[n * i + (k + 1)] = left_1;
                    B[n * j + (k + 1)] = right_1;
                    B[n * i + (k + 2)] = left_2;
                    B[n * j + (k + 2)] = right_2;
                    B[n * i + (k + 3)] = left_3;
                    B[n * j + (k + 3)] = right_3;
                    B[n * i + (k + 4)] = left_4;
                    B[n * j + (k + 4)] = right_4;
                    B[n * i + (k + 5)] = left_5;
                    B[n * j + (k + 5)] = right_5;
                    B[n * i + (k + 6)] = left_6;
                    B[n * j + (k + 6)] = right_6;
                    B[n * i + (k + 7)] = left_7;
                    B[n * j + (k + 7)] = right_7;
                }
                for (; k < n; ++k) {
                    double b_ik = B[n * i + k];
                    double b_jk = B[n * j + k];
                    double left = cf.c1 * b_ik - cf.s1 * b_jk;
                    double right = cf.s1 * cf.k * b_ik + cf.c1 * cf.k * b_jk;
                    B[n * i + k] = left;
                    B[n * j + k] = right;
                }

                k = 0;
                for (; k + 7 < n; k += 8) {
                    double b_ki_0 = B[n * (k + 0) + i];
                    double b_kj_0 = B[n * (k + 0) + j];
                    double b_ki_1 = B[n * (k + 1) + i];
                    double b_kj_1 = B[n * (k + 1) + j];
                    double b_ki_2 = B[n * (k + 2) + i];
                    double b_kj_2 = B[n * (k + 2) + j];
                    double b_ki_3 = B[n * (k + 3) + i];
                    double b_kj_3 = B[n * (k + 3) + j];
                    double b_ki_4 = B[n * (k + 4) + i];
                    double b_kj_4 = B[n * (k + 4) + j];
                    double b_ki_5 = B[n * (k + 5) + i];
                    double b_kj_5 = B[n * (k + 5) + j];
                    double b_ki_6 = B[n * (k + 6) + i];
                    double b_kj_6 = B[n * (k + 6) + j];
                    double b_ki_7 = B[n * (k + 7) + i];
                    double b_kj_7 = B[n * (k + 7) + j];

                    double left_0 = cf.c2 * b_ki_0 - cf.s2 * b_kj_0;
                    double right_0 = cf.s2 * b_ki_0 + cf.c2 * b_kj_0;
                    double left_1 = cf.c2 * b_ki_1 - cf.s2 * b_kj_1;
                    double right_1 = cf.s2 * b_ki_1 + cf.c2 * b_kj_1;
                    double left_2 = cf.c2 * b_ki_2 - cf.s2 * b_kj_2;
                    double right_2 = cf.s2 * b_ki_2 + cf.c2 * b_kj_2;
                    double left_3 = cf.c2 * b_ki_3 - cf.s2 * b_kj_3;
                    double right_3 = cf.s2 * b_ki_3 + cf.c2 * b_kj_3;
                    double left_4 = cf.c2 * b_ki_4 - cf.s2 * b_kj_4;
                    double right_4 = cf.s2 * b_ki_4 + cf.c2 * b_kj_4;
                    double left_5 = cf.c2 * b_ki_5 - cf.s2 * b_kj_5;
                    double right_5 = cf.s2 * b_ki_5 + cf.c2 * b_kj_5;
                    double left_6 = cf.c2 * b_ki_6 - cf.s2 * b_kj_6;
                    double right_6 = cf.s2 * b_ki_6 + cf.c2 * b_kj_6;
                    double left_7 = cf.c2 * b_ki_7 - cf.s2 * b_kj_7;
                    double right_7 = cf.s2 * b_ki_7 + cf.c2 * b_kj_7;

                    B[n * (k + 0) + i] = left_0;
                    B[n * (k + 0) + j] = right_0;
                    B[n * (k + 1) + i] = left_1;
                    B[n * (k + 1) + j] = right_1;
                    B[n * (k + 2) + i] = left_2;
                    B[n * (k + 2) + j] = right_2;
                    B[n * (k + 3) + i] = left_3;
                    B[n * (k + 3) + j] = right_3;
                    B[n * (k + 4) + i] = left_4;
                    B[n * (k + 4) + j] = right_4;
                    B[n * (k + 5) + i] = left_5;
                    B[n * (k + 5) + j] = right_5;
                    B[n * (k + 6) + i] = left_6;
                    B[n * (k + 6) + j] = right_6;
                    B[n * (k + 7) + i] = left_7;
                    B[n * (k + 7) + j] = right_7;
                }
                for (; k < n; ++k) {
                    double b_ki = B[n * k + i];
                    double b_kj = B[n * k + j];
                    double left = cf.c2 * b_ki - cf.s2 * b_kj;
                    double right = cf.s2 * b_ki + cf.c2 * b_kj;
                    B[n * k + i] = left;
                    B[n * k + j] = right;
                }

                k = 0;
                for (; k + 7 < n; k += 8) {
                    double u_ki_0 = U[n * (k + 0) + i];
                    double u_kj_0 = U[n * (k + 0) + j];
                    double u_ki_1 = U[n * (k + 1) + i];
                    double u_kj_1 = U[n * (k + 1) + j];
                    double u_ki_2 = U[n * (k + 2) + i];
                    double u_kj_2 = U[n * (k + 2) + j];
                    double u_ki_3 = U[n * (k + 3) + i];
                    double u_kj_3 = U[n * (k + 3) + j];
                    double u_ki_4 = U[n * (k + 4) + i];
                    double u_kj_4 = U[n * (k + 4) + j];
                    double u_ki_5 = U[n * (k + 5) + i];
                    double u_kj_5 = U[n * (k + 5) + j];
                    double u_ki_6 = U[n * (k + 6) + i];
                    double u_kj_6 = U[n * (k + 6) + j];
                    double u_ki_7 = U[n * (k + 7) + i];
                    double u_kj_7 = U[n * (k + 7) + j];

                    double left_0 = cf.c1 * u_ki_0 - cf.s1 * u_kj_0;
                    double right_0 = cf.s1 * cf.k * u_ki_0 + cf.c1 * cf.k * u_kj_0;
                    double left_1 = cf.c1 * u_ki_1 - cf.s1 * u_kj_1;
                    double right_1 = cf.s1 * cf.k * u_ki_1 + cf.c1 * cf.k * u_kj_1;
                    double left_2 = cf.c1 * u_ki_2 - cf.s1 * u_kj_2;
                    double right_2 = cf.s1 * cf.k * u_ki_2 + cf.c1 * cf.k * u_kj_2;
                    double left_3 = cf.c1 * u_ki_3 - cf.s1 * u_kj_3;
                    double right_3 = cf.s1 * cf.k * u_ki_3 + cf.c1 * cf.k * u_kj_3;
                    double left_4 = cf.c1 * u_ki_4 - cf.s1 * u_kj_4;
                    double right_4 = cf.s1 * cf.k * u_ki_4 + cf.c1 * cf.k * u_kj_4;
                    double left_5 = cf.c1 * u_ki_5 - cf.s1 * u_kj_5;
                    double right_5 = cf.s1 * cf.k * u_ki_5 + cf.c1 * cf.k * u_kj_5;
                    double left_6 = cf.c1 * u_ki_6 - cf.s1 * u_kj_6;
                    double right_6 = cf.s1 * cf.k * u_ki_6 + cf.c1 * cf.k * u_kj_6;
                    double left_7 = cf.c1 * u_ki_7 - cf.s1 * u_kj_7;
                    double right_7 = cf.s1 * cf.k * u_ki_7 + cf.c1 * cf.k * u_kj_7;

                    U[n * (k + 0) + i] = left_0;
                    U[n * (k + 0) + j] = right_0;
                    U[n * (k + 1) + i] = left_1;
                    U[n * (k + 1) + j] = right_1;
                    U[n * (k + 2) + i] = left_2;
                    U[n * (k + 2) + j] = right_2;
                    U[n * (k + 3) + i] = left_3;
                    U[n * (k + 3) + j] = right_3;
                    U[n * (k + 4) + i] = left_4;
                    U[n * (k + 4) + j] = right_4;
                    U[n * (k + 5) + i] = left_5;
                    U[n * (k + 5) + j] = right_5;
                    U[n * (k + 6) + i] = left_6;
                    U[n * (k + 6) + j] = right_6;
                    U[n * (k + 7) + i] = left_7;
                    U[n * (k + 7) + j] = right_7;
                }
                for (; k < n; ++k) {
                    double u_ki = U[n * k + i];
                    double u_kj = U[n * k + j];
                    double left = cf.c1 * u_ki - cf.s1 * u_kj;
                    double right = cf.s1 * cf.k * u_ki + cf.c1 * cf.k * u_kj;
                    U[n * k + i] = left;
                    U[n * k + j] = right;
                }

                k = 0;
                for (; k + 7 < n; k += 8) {
                    double v_ki_0 = V[n * (k + 0) + i];
                    double v_kj_0 = V[n * (k + 0) + j];
                    double v_ki_1 = V[n * (k + 1) + i];
                    double v_kj_1 = V[n * (k + 1) + j];
                    double v_ki_2 = V[n * (k + 2) + i];
                    double v_kj_2 = V[n * (k + 2) + j];
                    double v_ki_3 = V[n * (k + 3) + i];
                    double v_kj_3 = V[n * (k + 3) + j];
                    double v_ki_4 = V[n * (k + 4) + i];
                    double v_kj_4 = V[n * (k + 4) + j];
                    double v_ki_5 = V[n * (k + 5) + i];
                    double v_kj_5 = V[n * (k + 5) + j];
                    double v_ki_6 = V[n * (k + 6) + i];
                    double v_kj_6 = V[n * (k + 6) + j];
                    double v_ki_7 = V[n * (k + 7) + i];
                    double v_kj_7 = V[n * (k + 7) + j];

                    double left_0 = cf.c2 * v_ki_0 - cf.s2 * v_kj_0;
                    double right_0 = cf.s2 * v_ki_0 + cf.c2 * v_kj_0;
                    double left_1 = cf.c2 * v_ki_1 - cf.s2 * v_kj_1;
                    double right_1 = cf.s2 * v_ki_1 + cf.c2 * v_kj_1;
                    double left_2 = cf.c2 * v_ki_2 - cf.s2 * v_kj_2;
                    double right_2 = cf.s2 * v_ki_2 + cf.c2 * v_kj_2;
                    double left_3 = cf.c2 * v_ki_3 - cf.s2 * v_kj_3;
                    double right_3 = cf.s2 * v_ki_3 + cf.c2 * v_kj_3;
                    double left_4 = cf.c2 * v_ki_4 - cf.s2 * v_kj_4;
                    double right_4 = cf.s2 * v_ki_4 + cf.c2 * v_kj_4;
                    double left_5 = cf.c2 * v_ki_5 - cf.s2 * v_kj_5;
                    double right_5 = cf.s2 * v_ki_5 + cf.c2 * v_kj_5;
                    double left_6 = cf.c2 * v_ki_6 - cf.s2 * v_kj_6;
                    double right_6 = cf.s2 * v_ki_6 + cf.c2 * v_kj_6;
                    double left_7 = cf.c2 * v_ki_7 - cf.s2 * v_kj_7;
                    double right_7 = cf.s2 * v_ki_7 + cf.c2 * v_kj_7;

                    V[n * (k + 0) + i] = left_0;
                    V[n * (k + 0) + j] = right_0;
                    V[n * (k + 1) + i] = left_1;
                    V[n * (k + 1) + j] = right_1;
                    V[n * (k + 2) + i] = left_2;
                    V[n * (k + 2) + j] = right_2;
                    V[n * (k + 3) + i] = left_3;
                    V[n * (k + 3) + j] = right_3;
                    V[n * (k + 4) + i] = left_4;
                    V[n * (k + 4) + j] = right_4;
                    V[n * (k + 5) + i] = left_5;
                    V[n * (k + 5) + j] = right_5;
                    V[n * (k + 6) + i] = left_6;
                    V[n * (k + 6) + j] = right_6;
                    V[n * (k + 7) + i] = left_7;
                    V[n * (k + 7) + j] = right_7;
                }
                for (; k < n; ++k) {
                    double v_ki = V[n * k + i];
                    double v_kj = V[n * k + j];
                    double left = cf.c2 * v_ki - cf.s2 * v_kj;
                    double right = cf.s2 * v_ki + cf.c2 * v_kj;
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
