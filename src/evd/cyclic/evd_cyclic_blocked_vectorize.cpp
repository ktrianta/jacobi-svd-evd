#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include "block.hpp"
#include "evd_cost.hpp"
#include "evd_cyclic.hpp"
#include "matrix.hpp"
#include "nevd.hpp"
#include "types.hpp"
#include "util.hpp"

static inline size_t evd_block_vector(struct matrix_t Amat, struct matrix_t Vmat, size_t block_epoch);

size_t evd_cyclic_blocked_vectorize(struct matrix_t Data_matr, struct matrix_t Data_matr_copy,
                                    struct matrix_t Eigen_vectors, struct vector_t Eigen_values, int epoch,
                                    size_t block_epoch, const size_t block_size) {
    assert(Data_matr.rows == Data_matr.cols);  // Input Matrix should be square
    assert(Eigen_vectors.rows == Eigen_vectors.cols);
    struct matrix_t& Amat = Data_matr_copy;
    double* A = Amat.ptr;
    // Create a copy of the matrix to prevent modification of the original matrix
    memcpy(A, Data_matr.ptr, Data_matr.rows * Data_matr.cols * sizeof(double));

    double* E = Eigen_values.ptr;
    const size_t n = Amat.rows;
    const size_t n_blocks = n / block_size;
    size_t sub_cost = 0;
    matrix_identity(Eigen_vectors);

    if (n < 2 * block_size) {
        evd_block_vector(Amat, Eigen_vectors, block_epoch);

        // Store the generated eigen values in the vector
        for (size_t i = 0; i < n; i++) {
            E[i] = A[i * n + i];
        }
        reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
        return base_cost_evd(n, block_epoch);
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

                sub_cost += evd_block_vector(Ablockmat, Vblockmat, block_epoch);
                // Cant use this because our cost is wrong
                //sub_cost += evd_subprocedure_vectorized(Ablockmat, Vblockmat, block_epoch);

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
    return blocked_cost_without_subprocedure_evd(n, block_size, epoch) + sub_cost;
}

size_t evd_cyclic_blocked_less_copy_vectorize(struct matrix_t Data_matr, struct matrix_t Data_matr_copy,
                                              struct matrix_t Eigen_vectors, struct vector_t Eigen_values, int epoch,
                                              size_t block_epoch, const size_t block_size) {
    assert(Data_matr.rows == Data_matr.cols);  // Input Matrix should be square
    assert(Eigen_vectors.rows == Eigen_vectors.cols);
    struct matrix_t& Amat = Data_matr_copy;
    double* A = Amat.ptr;
    // Create a copy of the matrix to prevent modification of the original matrix
    memcpy(A, Data_matr.ptr, Data_matr.rows * Data_matr.cols * sizeof(double));
    size_t sub_cost = 0;
    double* E = Eigen_values.ptr;
    const size_t n = Amat.rows;
    const size_t n_blocks = n / block_size;

    matrix_identity(Eigen_vectors);

    if (n < 2 * block_size) {
        evd_block_vector(Amat, Eigen_vectors, block_epoch);

        // Store the generated eigen values in the vector
        for (size_t i = 0; i < n; i++) {
            E[i] = A[i * n + i];
        }
        reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
        return base_cost_evd(n, block_epoch);
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

                sub_cost += evd_block_vector(Ablockmat, Vblockmat, block_epoch);
                // We have the wrong cost for this.
                //sub_cost += evd_subprocedure_vectorized(Ablockmat, Vblockmat, block_epoch);

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

    return blocked_less_copy_cost_without_subprocedure_evd(n, block_size, epoch) + sub_cost;
}

// Perform EVD for the block
static size_t evd_block_vector(struct matrix_t Amat, struct matrix_t Vmat, size_t epoch) {
    assert(Amat.rows == Amat.cols);
    double* A = Amat.ptr;

    double* V = Vmat.ptr;
    const size_t m = Amat.rows;

    matrix_identity(Vmat);

    for (size_t ep = 1; ep <= epoch; ep++) {
        double cos_t, sin_t;

        for (size_t row = 0; row < m; row++) {
            for (size_t col = row + 1; col < m; col++) {
                size_t n = m;
                __m256d sin_vec, cos_vec;

                // Compute cos_t and sin_t for the rotation
                sym_jacobi_coeffs(A[row * m + row], A[row * m + col], A[col * m + col], &cos_t, &sin_t);

                sin_vec = _mm256_set1_pd(sin_t);
                cos_vec = _mm256_set1_pd(cos_t);

                if (m % 4 != 0) n = m - (m % 4);

                for (size_t i = 0; i < n; i += 4) {
                    __m256d Ac_row, Ac_col, Ac_rcopy;
                    __m256d sin_row, sin_col, cos_row, cos_col;

                    // Compute the eigen values by updating the columns until convergence
                    Ac_row = _mm256_set_pd(A[m * i + row], A[m * i + m + row], A[m * i + m * 2 + row],
                                           A[m * i + m * 3 + row]);
                    Ac_rcopy = Ac_row;
                    Ac_col = _mm256_set_pd(A[m * i + col], A[m * i + m + col], A[m * i + m * 2 + col],
                                           A[m * i + m * 3 + col]);

                    cos_row = _mm256_mul_pd(Ac_row, cos_vec);
                    sin_col = _mm256_mul_pd(Ac_col, sin_vec);
                    Ac_row = _mm256_sub_pd(cos_row, sin_col);
                    double* Ac_row_updated = (double*) &Ac_row;
                    A[m * i + row] = Ac_row_updated[3];
                    A[m * i + m + row] = Ac_row_updated[2];
                    A[m * i + m * 2 + row] = Ac_row_updated[1];
                    A[m * i + m * 3 + row] = Ac_row_updated[0];

                    cos_col = _mm256_mul_pd(Ac_col, cos_vec);
                    sin_row = _mm256_mul_pd(Ac_rcopy, sin_vec);
                    Ac_col = _mm256_add_pd(cos_col, sin_row);
                    double* Ac_col_updated = (double*) &Ac_col;
                    A[m * i + col] = Ac_col_updated[3];
                    A[m * i + m + col] = Ac_col_updated[2];
                    A[m * i + m * 2 + col] = Ac_col_updated[1];
                    A[m * i + m * 3 + col] = Ac_col_updated[0];
                }

                if (m % 4 != 0) {
                    for (size_t i = 0; i < m - n; i++) {
                        double A_i_r = A[m * (n + i) + row];
                        A[m * (n + i) + row] = cos_t * A[m * (n + i) + row] - sin_t * A[m * (n + i) + col];
                        A[m * (n + i) + col] = cos_t * A[m * (n + i) + col] + sin_t * A_i_r;
                    }
                }

                for (size_t i = 0; i < n; i += 4) {
                    __m256d A_row, A_col, A_rcopy, V_row, V_col, V_rcopy;
                    __m256d sin_row, sin_col, cos_row, cos_col;

                    // Compute the eigen values by updating the rows until convergence
                    A_row = _mm256_loadu_pd(A + m * row + i);
                    A_rcopy = A_row;
                    A_col = _mm256_loadu_pd(A + m * col + i);

                    cos_row = _mm256_mul_pd(A_row, cos_vec);
                    sin_col = _mm256_mul_pd(A_col, sin_vec);
                    A_row = _mm256_sub_pd(cos_row, sin_col);
                    _mm256_storeu_pd(A + m * row + i, A_row);

                    cos_col = _mm256_mul_pd(A_col, cos_vec);
                    sin_row = _mm256_mul_pd(A_rcopy, sin_vec);
                    A_col = _mm256_add_pd(cos_col, sin_row);
                    _mm256_storeu_pd(A + m * col + i, A_col);

                    // Compute the eigen vectors similarly by updating the eigen vector matrix
                    V_row = _mm256_loadu_pd(V + m * row + i);
                    V_rcopy = V_row;
                    V_col = _mm256_loadu_pd(V + m * col + i);

                    cos_row = _mm256_mul_pd(V_row, cos_vec);
                    sin_col = _mm256_mul_pd(V_col, sin_vec);
                    V_row = _mm256_sub_pd(cos_row, sin_col);
                    _mm256_storeu_pd(V + m * row + i, V_row);

                    cos_col = _mm256_mul_pd(V_col, cos_vec);
                    sin_row = _mm256_mul_pd(V_rcopy, sin_vec);
                    V_col = _mm256_add_pd(cos_col, sin_row);
                    _mm256_storeu_pd(V + m * col + i, V_col);
                }

                if (m % 4 != 0) {
                    for (size_t i = 0; i < m - n; i++) {
                        double A_r_i = A[m * row + n + i];
                        A[m * row + n + i] = cos_t * A[m * row + n + i] - sin_t * A[m * col + n + i];
                        A[m * col + n + i] = cos_t * A[m * col + n + i] + sin_t * A_r_i;

                        double V_r_i = V[m * row + n + i];
                        V[m * row + n + i] = cos_t * V[m * row + n + i] - sin_t * V[m * col + n + i];
                        V[m * col + n + i] = cos_t * V[m * col + n + i] + sin_t * V_r_i;
                    }
                }
            }
        }
    }

    matrix_transpose({V, m, m}, {V, m, m});
    return base_cost_evd(m,epoch);
}
