#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include "block.hpp"
#include "evd_cost.hpp"
#include "evd_cyclic.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include "util.hpp"

static inline size_t evd_block(struct matrix_t Amat, struct matrix_t Vmat, size_t block_epoch);

size_t evd_cyclic_blocked_unroll_outer(struct matrix_t Data_matr, struct matrix_t Data_matr_copy,
                                       struct matrix_t Eigen_vectors, struct vector_t Eigen_values, int epoch,
                                       size_t block_epoch, size_t block_size) {
    assert(Data_matr.rows == Data_matr.cols);  // Input Matrix should be square
    assert(Eigen_vectors.rows == Eigen_vectors.cols);
    struct matrix_t& Amat = Data_matr_copy;
    double* A = Amat.ptr;
    // Create a copy of the matrix to prevent modification of the original matrix
    memcpy(A, Data_matr.ptr, Data_matr.rows * Data_matr.cols * sizeof(double));

    double* E = Eigen_values.ptr;
    const size_t n = Amat.rows;
    const size_t n_blocks = n / block_size;

    matrix_identity(Eigen_vectors);

    if (n <= 2 * block_size) {
        evd_block(Amat, Eigen_vectors, block_epoch);

        // Store the generated eigen values in the vector
        for (size_t i = 0; i < n; i++) {
            E[i] = A[i * n + i];
        }
        reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
        return oneloop_cost_evd(n, block_epoch);
    }

    assert(n_blocks * block_size == n);

    double* memory_block = (double*) aligned_alloc(32, (4 + 4 + 1 + 1) * block_size * block_size * sizeof(double));
    double* Ablock = memory_block;
    double* Vblock = Ablock + 4 * block_size * block_size;
    double* M1 = Vblock + 4 * block_size * block_size;
    double* M2 = M1 + block_size * block_size;

    matrix_t Ablockmat = {Ablock, 2 * block_size, 2 * block_size};
    matrix_t Vblockmat = {Vblock, 2 * block_size, 2 * block_size};
    matrix_t M1mat = {M1, block_size, block_size};
    matrix_t M2mat = {M2, block_size, block_size};

    for (int ep = 1; ep <= epoch; ep++) {
        for (size_t i_block = 0; i_block < n_blocks - 1; ++i_block) {
            for (size_t j_block = i_block + 1; j_block < n_blocks; ++j_block) {
                copy_block(Amat, i_block, i_block, Ablockmat, 0, 0, block_size);
                copy_block(Amat, i_block, j_block, Ablockmat, 0, 1, block_size);
                copy_block(Amat, j_block, i_block, Ablockmat, 1, 0, block_size);
                copy_block(Amat, j_block, j_block, Ablockmat, 1, 1, block_size);

                evd_block(Ablockmat, Vblockmat, block_epoch);

                matrix_transpose(Vblockmat, Vblockmat);

                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Vblockmat, 0, 0, Amat, i_block, k_block, M1mat, 0, 0, block_size);
                    mult_block(Vblockmat, 0, 1, Amat, j_block, k_block, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    mult_block(Vblockmat, 1, 0, Amat, i_block, k_block, M1mat, 0, 0, block_size);
                    copy_block(M2mat, 0, 0, Amat, i_block, k_block, block_size);
                    mult_block(Vblockmat, 1, 1, Amat, j_block, k_block, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    copy_block(M2mat, 0, 0, Amat, j_block, k_block, block_size);
                }
                matrix_transpose(Vblockmat, Vblockmat);

                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Amat, k_block, i_block, Vblockmat, 0, 0, M1mat, 0, 0, block_size);
                    mult_block(Amat, k_block, j_block, Vblockmat, 1, 0, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    mult_block(Amat, k_block, i_block, Vblockmat, 0, 1, M1mat, 0, 0, block_size);
                    copy_block(M2mat, 0, 0, Amat, k_block, i_block, block_size);
                    mult_block(Amat, k_block, j_block, Vblockmat, 1, 1, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    copy_block(M2mat, 0, 0, Amat, k_block, j_block, block_size);
                }

                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Eigen_vectors, k_block, i_block, Vblockmat, 0, 0, M1mat, 0, 0, block_size);
                    mult_block(Eigen_vectors, k_block, j_block, Vblockmat, 1, 0, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    mult_block(Eigen_vectors, k_block, i_block, Vblockmat, 0, 1, M1mat, 0, 0, block_size);
                    copy_block(M2mat, 0, 0, Eigen_vectors, k_block, i_block, block_size);
                    mult_block(Eigen_vectors, k_block, j_block, Vblockmat, 1, 1, M2mat, 0, 0, block_size);
                    matrix_add(M1mat, M2mat, M2mat);
                    copy_block(M2mat, 0, 0, Eigen_vectors, k_block, j_block, block_size);
                }
            }
        }
    }

    free(memory_block);
    // Store the generated eigen values in the vector
    for (size_t i = 0; i < n; i++) {
        E[i] = A[i * n + i];
    }
    reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
    return blocked_cost_without_subprocedure_evd(n, block_size, epoch) +
           epoch * n_blocks * (n_blocks - 1) * 0.5 * oneloop_cost_evd(2 * block_size, block_epoch);
}  // NOLINT

size_t evd_cyclic_blocked_unroll_outer_less_copy(struct matrix_t Data_matr, struct matrix_t Data_matr_copy,
                                                 struct matrix_t Eigen_vectors, struct vector_t Eigen_values, int epoch,
                                                 size_t block_epoch, size_t block_size) {
    assert(Data_matr.rows == Data_matr.cols);  // Input Matrix should be square
    assert(Eigen_vectors.rows == Eigen_vectors.cols);
    struct matrix_t& Amat = Data_matr_copy;
    double* A = Amat.ptr;
    // Create a copy of the matrix to prevent modification of the original matrix
    memcpy(A, Data_matr.ptr, Data_matr.rows * Data_matr.cols * sizeof(double));

    double* E = Eigen_values.ptr;
    const size_t n = Amat.rows;
    const size_t n_blocks = n / block_size;
    size_t block_cost = 0;

    matrix_identity(Eigen_vectors);

    if (n <= 2 * block_size) {
        evd_block(Amat, Eigen_vectors, block_epoch);

        // Store the generated eigen values in the vector
        for (size_t i = 0; i < n; i++) {
            E[i] = A[i * n + i];
        }
        reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
        return oneloop_cost_evd(n, block_epoch);
    }

    assert(n_blocks * block_size == n);

    double* memory_block = (double*) aligned_alloc(32, (4 + 4 + 1 + 1) * block_size * block_size * sizeof(double));
    double* Ablock = memory_block;
    double* Vblock = Ablock + 4 * block_size * block_size;
    double* M1 = Vblock + 4 * block_size * block_size;
    double* M2 = M1 + block_size * block_size;

    matrix_t Ablockmat = {Ablock, 2 * block_size, 2 * block_size};
    matrix_t Vblockmat = {Vblock, 2 * block_size, 2 * block_size};
    matrix_t M1mat = {M1, block_size, block_size};
    matrix_t M2mat = {M2, block_size, block_size};

    for (int ep = 1; ep <= epoch; ep++) {
        for (size_t i_block = 0; i_block < n_blocks - 1; ++i_block) {
            for (size_t j_block = i_block + 1; j_block < n_blocks; ++j_block) {
                copy_block(Amat, i_block, i_block, Ablockmat, 0, 0, block_size);
                copy_block(Amat, i_block, j_block, Ablockmat, 0, 1, block_size);
                copy_block(Amat, j_block, i_block, Ablockmat, 1, 0, block_size);
                copy_block(Amat, j_block, j_block, Ablockmat, 1, 1, block_size);

                block_cost += evd_block(Ablockmat, Vblockmat, block_epoch);

                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_transpose_block(Vblockmat, 0, 0, Amat, i_block, k_block, M1mat, 0, 0, block_size);
                    mult_transpose_block(Vblockmat, 0, 1, Amat, i_block, k_block, M2mat, 0, 0, block_size);
                    mult_add_transpose_block(Vblockmat, 1, 0, Amat, j_block, k_block, M1mat, 0, 0, Amat, i_block,
                                             k_block, block_size);
                    copy_block(Amat, j_block, k_block, M1mat, 0, 0, block_size);
                    mult_add_transpose_block(Vblockmat, 1, 1, M1mat, 0, 0, M2mat, 0, 0, Amat, j_block, k_block,
                                             block_size);
                }

                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Amat, k_block, i_block, Vblockmat, 0, 0, M1mat, 0, 0, block_size);
                    mult_block(Amat, k_block, i_block, Vblockmat, 0, 1, M2mat, 0, 0, block_size);

                    mult_add_block(Amat, k_block, j_block, Vblockmat, 1, 0, M1mat, 0, 0, Amat, k_block, i_block,
                                   block_size);
                    copy_block(Amat, k_block, j_block, M1mat, 0, 0, block_size);
                    mult_add_block(M1mat, 0, 0, Vblockmat, 1, 1, M2mat, 0, 0, Amat, k_block, j_block, block_size);
                }

                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Eigen_vectors, k_block, i_block, Vblockmat, 0, 0, M1mat, 0, 0, block_size);
                    mult_block(Eigen_vectors, k_block, i_block, Vblockmat, 0, 1, M2mat, 0, 0, block_size);
                    mult_add_block(Eigen_vectors, k_block, j_block, Vblockmat, 1, 0, M1mat, 0, 0, Eigen_vectors,
                                   k_block, i_block, block_size);
                    copy_block(Eigen_vectors, k_block, j_block, M1mat, 0, 0, block_size);
                    mult_add_block(M1mat, 0, 0, Vblockmat, 1, 1, M2mat, 0, 0, Eigen_vectors, k_block, j_block,
                                   block_size);
                }
            }
        }
    }

    free(memory_block);
    // Store the generated eigen values in the vector
    for (size_t i = 0; i < n; i++) {
        E[i] = A[i * n + i];
    }
    reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
    return blocked_less_copy_cost_without_subprocedure_evd(n, block_size, epoch) + block_cost;
}

size_t evd_block(struct matrix_t Data_matr, struct matrix_t Eigen_vectors, size_t block_epoch) {
    assert(Data_matr.rows == Data_matr.cols);
    double* A = Data_matr.ptr;
    double* V = Eigen_vectors.ptr;
    size_t n = Data_matr.rows;

    double c0, s0;
    double c1, s1;

    matrix_identity(Eigen_vectors);

    for (size_t ep = 0; ep < block_epoch; ++ep) {
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
    return oneloop_cost_evd(n, block_epoch);
}
