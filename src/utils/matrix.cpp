#include "matrix.hpp"
#include <cassert>
#include <utility>

void matrix_identity(matrix_t Pmat) {
    double* P = Pmat.ptr;
    const size_t M = Pmat.rows;
    const size_t N = Pmat.cols;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            if (i == j) {
                P[i * N + j] = 1.0;
            } else {
                P[i * N + j] = 0.0;
            }
        }
    }
}

void matrix_transpose(matrix_t Pmat, matrix_t Qmat) {
    double* P = Pmat.ptr;
    double* Q = Qmat.ptr;
    const size_t M = Pmat.rows;
    const size_t N = Pmat.cols;

    // Swap for in-space transpose
    if (P == Q) {
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < i; j++) {
                std::swap(Q[i * N + j], Q[j * N + i]);
            }
        }
    } else {
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                Q[j * N + i] = P[i * N + j];
            }
        }
    }
}

void matrix_mult(matrix_t Pmat, matrix_t Qmat, matrix_t Rmat) {
    double* P = Pmat.ptr;
    double* Q = Qmat.ptr;
    double* R = Rmat.ptr;
    const size_t M = Pmat.rows;
    const size_t N = Pmat.cols;
    const size_t O = Qmat.cols;

    assert(Qmat.cols == Rmat.rows);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            double sum = 0.0;

            for (size_t k = 0; k < O; k++) {
                sum += Q[i * N + k] * R[k * N + j];
            }

            P[i * N + j] = sum;
        }
    }
}
