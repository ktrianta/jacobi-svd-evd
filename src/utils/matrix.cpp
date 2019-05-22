#include "matrix.hpp"
#include <immintrin.h>
#include <algorithm>
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

void matrix_add(matrix_t Amat, matrix_t Bmat, matrix_t Cmat) {
    size_t n = Amat.rows;
    double* A = Amat.ptr;
    double* B = Bmat.ptr;
    double* C = Cmat.ptr;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            C[n * i + j] = A[n * i + j] + B[n * i + j];
        }
    }
}

void matrix_copy(matrix_t Bmat, matrix_t Amat) {
    assert(Amat.rows == Bmat.rows);
    assert(Amat.cols == Amat.cols);

    std::copy(Amat.ptr, Amat.ptr + Amat.rows * Amat.cols, Bmat.ptr);
}

void matrix_frobenius(matrix_t m, double* norm, double* off_norm) {
    const size_t M = m.rows;
    const size_t N = m.cols;
    double* data = m.ptr;
    double elems_sum = 0.0;           // sum m[i][j]^2 for 0 < i < M and 0 < j < N
    double off_diag_elems_sum = 0.0;  // sum m[i][j]^2 for 0 < i < M and 0 < j < N and i == j

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            elems_sum += data[N * i + j] * data[N * i + j];

            if (i == j) {
                continue;
            } else {
                off_diag_elems_sum += data[N * i + j] * data[N * i + j];
            }
        }
    }

    *norm = elems_sum;
    *off_norm = off_diag_elems_sum;
}

void matrix_frobenius_vectorized(matrix_t m, double* norm, double* off_norm) {
    const size_t M = m.rows;
    const size_t N = m.cols;
    const size_t N4 = N - (N % 4);
    double* data = m.ptr;
    double off_diag_elems_sum, rem = 0.0;  // sum m[i][j]^2 for 0 < i < M and 0 < j < N and i == j
    __m256d mvals, squared_mvals, all_sum, rem_sum;

    all_sum = _mm256_set1_pd(0.0);
    rem_sum = _mm256_set1_pd(0.0);
    for (size_t i = 0; i < M; ++i) {
        size_t j = 0;
        for (; j < N4; j += 4) {
            if ((j == i) || (j + 1 == i) || (j + 2 == i) || (j + 3 == i)) {
                if (j == i) {
                    rem += data[N * i + j + 1] * data[N * i + j + 1];
                    rem += data[N * i + j + 2] * data[N * i + j + 2];
                    rem += data[N * i + j + 3] * data[N * i + j + 3];
                } else if (j == i + 1) {
                    rem += data[N * i + j] * data[N * i + j];
                    rem += data[N * i + j + 2] * data[N * i + j + 2];
                    rem += data[N * i + j + 3] * data[N * i + j + 3];
                } else if (j == i + 2) {
                    rem += data[N * i + j] * data[N * i + j];
                    rem += data[N * i + j + 1] * data[N * i + j + 1];
                    rem += data[N * i + j + 3] * data[N * i + j + 3];
                } else if (j == i + 3) {
                    rem += data[N * i + j] * data[N * i + j];
                    rem += data[N * i + j + 1] * data[N * i + j + 1];
                    rem += data[N * i + j + 2] * data[N * i + j + 2];
                }
                mvals = _mm256_loadu_pd(data + N * i + j);
                squared_mvals = _mm256_mul_pd(mvals, mvals);
                all_sum = _mm256_add_pd(squared_mvals, all_sum);
            } else {
                mvals = _mm256_loadu_pd(data + N * i + j);
                squared_mvals = _mm256_mul_pd(mvals, mvals);
                rem_sum = _mm256_add_pd(squared_mvals, rem_sum);
            }
        }

        for (; j < N; j++) {
            if (j != i) {
                rem += data[N * i + j] * data[N * i + j];
            }
        }
    }

    off_diag_elems_sum = all_sum[0] + all_sum[1] + all_sum[2] + all_sum[3] + rem;
    *off_norm = off_diag_elems_sum;
    *norm = off_diag_elems_sum + rem_sum[0] + rem_sum[1] + rem_sum[2] + rem_sum[3];
}

void matrix_off_frobenius(matrix_t m, double* off_norm) {
    const size_t M = m.rows;
    const size_t N = m.cols;
    double* data = m.ptr;
    double off_diag_elems_sum = 0.0;  // sum m[i][j]^2 for 0 < i < M and 0 < j < N and i == j

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i == j) {
                continue;
            } else {
                off_diag_elems_sum += data[N * i + j] * data[N * i + j];
            }
        }
    }

    *off_norm = off_diag_elems_sum;
}

void matrix_off_frobenius_vectorized(matrix_t m, double* off_norm) {
    const size_t M = m.rows;
    const size_t N = m.cols;
    const size_t N4 = (N > N % 4 ? N - (N % 4) : N);
    double* data = m.ptr;
    double off_diag_elems_sum = 0.0, rem = 0.0;  // sum m[i][j]^2 for 0 < i < M and 0 < j < N and i == j
    __m256d mvals, squared_mvals, all_sum;

    all_sum = _mm256_set1_pd(0.0);
    for (size_t i = 0; i < M; ++i) {
        size_t j = 0;
        for (; j < N4; j += 4) {
            if ((j == i) || (j + 1 == i) || (j + 2 == i) || (j + 3 == i)) {
                if (j == i) {
                    rem += data[N * i + j + 1] * data[N * i + j + 1];
                    rem += data[N * i + j + 2] * data[N * i + j + 2];
                    rem += data[N * i + j + 3] * data[N * i + j + 3];
                } else if (j == i + 1) {
                    rem += data[N * i + j] * data[N * i + j];
                    rem += data[N * i + j + 2] * data[N * i + j + 2];
                    rem += data[N * i + j + 3] * data[N * i + j + 3];
                } else if (j == i + 2) {
                    rem += data[N * i + j] * data[N * i + j];
                    rem += data[N * i + j + 1] * data[N * i + j + 1];
                    rem += data[N * i + j + 3] * data[N * i + j + 3];
                } else if (j == i + 3) {
                    rem += data[N * i + j] * data[N * i + j];
                    rem += data[N * i + j + 1] * data[N * i + j + 1];
                    rem += data[N * i + j + 2] * data[N * i + j + 2];
                }
            } else {
                mvals = _mm256_loadu_pd(data + N * i + j);
                squared_mvals = _mm256_mul_pd(mvals, mvals);
                all_sum = _mm256_add_pd(squared_mvals, all_sum);
            }
        }

        for (; j < N; j++) {
            if (j != i) {
                rem += data[N * i + j] * data[N * i + j];
            }
        }
    }

    off_diag_elems_sum = all_sum[0] + all_sum[1] + all_sum[2] + all_sum[3] + rem;
    *off_norm = off_diag_elems_sum;
}
