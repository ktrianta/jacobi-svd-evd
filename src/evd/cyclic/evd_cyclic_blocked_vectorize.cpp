#include "evd_cyclic.hpp"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include "matrix.hpp"
#include "types.hpp"
#include "util.hpp"

static inline void evd_block_vector(struct matrix_t Amat, struct matrix_t Vmat);
static inline void mult_block(struct matrix_t Amat, size_t blockA_row, size_t blockA_col, struct matrix_t Bmat,
                              size_t blockB_row, size_t blockB_col, struct matrix_t Cmat, size_t blockC_row,
                              size_t blockC_col, size_t block_size);
static inline void add(struct matrix_t Amat, struct matrix_t Bmat, struct matrix_t Cmat);
static inline void copy_block(struct matrix_t S, size_t blockS_row, size_t blockS_col, struct matrix_t D,
                              size_t blockD_row, size_t blockD_col, size_t block_size);
static inline void transpose(struct matrix_t A);
static inline void mult_add_block(struct matrix_t Amat, size_t blockA_row, size_t blockA_col, struct matrix_t Bmat,
                    size_t blockB_row, size_t blockB_col, struct matrix_t Cmat, size_t blockC_row, size_t blockC_col,
                    struct matrix_t Dmat, size_t blockD_row, size_t blockD_col, size_t block_size);
static inline void mult_transpose_block(struct matrix_t Amat, size_t blockA_row, size_t blockA_col,
                          struct matrix_t Bmat, size_t blockB_row, size_t blockB_col, struct matrix_t Cmat,
                          size_t blockC_row, size_t blockC_col, size_t block_size);
static inline void mult_add_transpose_block(struct matrix_t Amat, size_t blockA_row, size_t blockA_col,
                          struct matrix_t Bmat, size_t blockB_row, size_t blockB_col, struct matrix_t Cmat,
                          size_t blockC_row, size_t blockC_col, struct matrix_t Dmat, size_t blockD_row,
                          size_t blockD_col, size_t block_size);

void evd_cyclic_blocked_vectorize(struct matrix_t Data_matr, struct matrix_t Data_matr_copy,
            struct matrix_t Eigen_vectors, struct vector_t Eigen_values, int epoch) {
    assert(Data_matr.rows == Data_matr.cols);  // Input Matrix should be square
    assert(Eigen_vectors.rows == Eigen_vectors.cols);
    struct matrix_t& Amat = Data_matr_copy;
    double* A = Amat.ptr;
    // Create a copy of the matrix to prevent modification of the original matrix
    memcpy(A, Data_matr.ptr, Data_matr.rows * Data_matr.cols * sizeof(double));

    double* E = Eigen_values.ptr;
    const size_t n = Amat.rows;
    const size_t block_size = 8;
    const size_t n_blocks = n / block_size;

    matrix_identity(Eigen_vectors);

    int is_not_diagonal = 0;

    if (n < 2 * block_size) {
        evd_block_vector(Amat, Eigen_vectors);

        // Store the generated eigen values in the vector
        for (size_t i = 0; i < n; i++) {
            E[i] = A[i * n + i];
        }
        reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
        return;
    }

    assert(n_blocks * block_size == n);

    double* memory_block = (double*) malloc((4 + 4 + 1 + 1) * block_size * block_size * sizeof(double));
    double* Ablock = memory_block;
    double* Vblock = Ablock + 4 * block_size * block_size;
    double* M1 = Vblock + 4 * block_size * block_size;
    double* M2 = M1 + block_size * block_size;

    matrix_t Ablockmat = {Ablock, 2 * block_size, 2 * block_size};
    matrix_t Vblockmat = {Vblock, 2 * block_size, 2 * block_size};
    matrix_t M1mat = {M1, block_size, block_size};
    matrix_t M2mat = {M2, block_size, block_size};

    for (int ep = 1; ep <= epoch; ep++) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i + 1; j < n; j++) {
                if (A[i * n + j] != 0.0) {
                    is_not_diagonal = 1;
                    break;
                }
            }
        }

        if (!is_not_diagonal) break;

        for (size_t i_block = 0; i_block < n_blocks - 1; ++i_block) {
            for (size_t j_block = i_block + 1; j_block < n_blocks; ++j_block) {
                copy_block(Amat, i_block, i_block, Ablockmat, 0, 0, block_size);
                copy_block(Amat, i_block, j_block, Ablockmat, 0, 1, block_size);
                copy_block(Amat, j_block, i_block, Ablockmat, 1, 0, block_size);
                copy_block(Amat, j_block, j_block, Ablockmat, 1, 1, block_size);

                evd_block_vector(Ablockmat, Vblockmat);

                transpose(Vblockmat);

                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Vblockmat, 0, 0, Amat, i_block, k_block, M1mat, 0, 0, block_size);
                    mult_block(Vblockmat, 0, 1, Amat, j_block, k_block, M2mat, 0, 0, block_size);
                    add(M1mat, M2mat, M2mat);
                    mult_block(Vblockmat, 1, 0, Amat, i_block, k_block, M1mat, 0, 0, block_size);
                    copy_block(M2mat, 0, 0, Amat, i_block, k_block, block_size);
                    mult_block(Vblockmat, 1, 1, Amat, j_block, k_block, M2mat, 0, 0, block_size);
                    add(M1mat, M2mat, M2mat);
                    copy_block(M2mat, 0, 0, Amat, j_block, k_block, block_size);
                }
                transpose(Vblockmat);

                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Amat, k_block, i_block, Vblockmat, 0, 0, M1mat, 0, 0, block_size);
                    mult_block(Amat, k_block, j_block, Vblockmat, 1, 0, M2mat, 0, 0, block_size);
                    add(M1mat, M2mat, M2mat);
                    mult_block(Amat, k_block, i_block, Vblockmat, 0, 1, M1mat, 0, 0, block_size);
                    copy_block(M2mat, 0, 0, Amat, k_block, i_block, block_size);
                    mult_block(Amat, k_block, j_block, Vblockmat, 1, 1, M2mat, 0, 0, block_size);
                    add(M1mat, M2mat, M2mat);
                    copy_block(M2mat, 0, 0, Amat, k_block, j_block, block_size);
                }

                for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
                    mult_block(Eigen_vectors, k_block, i_block, Vblockmat, 0, 0, M1mat, 0, 0, block_size);
                    mult_block(Eigen_vectors, k_block, j_block, Vblockmat, 1, 0, M2mat, 0, 0, block_size);
                    add(M1mat, M2mat, M2mat);
                    mult_block(Eigen_vectors, k_block, i_block, Vblockmat, 0, 1, M1mat, 0, 0, block_size);
                    copy_block(M2mat, 0, 0, Eigen_vectors, k_block, i_block, block_size);
                    mult_block(Eigen_vectors, k_block, j_block, Vblockmat, 1, 1, M2mat, 0, 0, block_size);
                    add(M1mat, M2mat, M2mat);
                    copy_block(M2mat, 0, 0, Eigen_vectors, k_block, j_block, block_size);
                }
            }
        }
        is_not_diagonal = 0;
    }

    free(memory_block);
    // Store the generated eigen values in the vector
    for (size_t i = 0; i < n; i++) {
        E[i] = A[i * n + i];
    }
    reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
}

void evd_cyclic_blocked_less_copy_vectorize(struct matrix_t Data_matr, struct matrix_t Data_matr_copy,
                    struct matrix_t Eigen_vectors, struct vector_t Eigen_values, int epoch) {
    assert(Data_matr.rows == Data_matr.cols);  // Input Matrix should be square
    assert(Eigen_vectors.rows == Eigen_vectors.cols);
    struct matrix_t& Amat = Data_matr_copy;
    double* A = Amat.ptr;
    // Create a copy of the matrix to prevent modification of the original matrix
    memcpy(A, Data_matr.ptr, Data_matr.rows * Data_matr.cols * sizeof(double));

    double* E = Eigen_values.ptr;
    const size_t n = Amat.rows;
    const size_t block_size = 8;
    const size_t n_blocks = n / block_size;

    matrix_identity(Eigen_vectors);

    int is_not_diagonal = 0;

    if (n < 2 * block_size) {
        evd_block_vector(Amat, Eigen_vectors);

        // Store the generated eigen values in the vector
        for (size_t i = 0; i < n; i++) {
            E[i] = A[i * n + i];
        }
        reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
        return;
    }

    assert(n_blocks * block_size == n);

    double* memory_block = (double*) malloc((4 + 4 + 1 + 1) * block_size * block_size * sizeof(double));
    double* Ablock = memory_block;
    double* Vblock = Ablock + 4 * block_size * block_size;
    double* M1 = Vblock + 4 * block_size * block_size;
    double* M2 = M1 + block_size * block_size;

    matrix_t Ablockmat = {Ablock, 2 * block_size, 2 * block_size};
    matrix_t Vblockmat = {Vblock, 2 * block_size, 2 * block_size};
    matrix_t M1mat = {M1, block_size, block_size};
    matrix_t M2mat = {M2, block_size, block_size};

    for (int ep = 1; ep <= epoch; ep++) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i + 1; j < n; j++) {
                if (A[i * n + j] != 0.0) {
                    is_not_diagonal = 1;
                    break;
                }
            }
        }

        if (!is_not_diagonal) break;

        for (size_t i_block = 0; i_block < n_blocks - 1; ++i_block) {
            for (size_t j_block = i_block + 1; j_block < n_blocks; ++j_block) {
                copy_block(Amat, i_block, i_block, Ablockmat, 0, 0, block_size);
                copy_block(Amat, i_block, j_block, Ablockmat, 0, 1, block_size);
                copy_block(Amat, j_block, i_block, Ablockmat, 1, 0, block_size);
                copy_block(Amat, j_block, j_block, Ablockmat, 1, 1, block_size);

                evd_block_vector(Ablockmat, Vblockmat);

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
                    mult_add_block(Eigen_vectors, k_block, j_block, Vblockmat, 1, 0, M1mat, 0, 0,
                                   Eigen_vectors, k_block, i_block, block_size);
                    copy_block(Eigen_vectors, k_block, j_block, M1mat, 0, 0, block_size);
                    mult_add_block(M1mat, 0, 0, Vblockmat, 1, 1, M2mat, 0, 0, Eigen_vectors, k_block,
                                   j_block, block_size);
                }
            }
        }
        is_not_diagonal = 0;
    }

    free(memory_block);
    // Store the generated eigen values in the vector
    for (size_t i = 0; i < n; i++) {
        E[i] = A[i * n + i];
    }
    reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
}

// Perform EVD for the block
static void evd_block_vector(struct matrix_t Amat, struct matrix_t Vmat) {
  assert(Amat.rows == Amat.cols);
  double* A = Amat.ptr;

  double* V = Vmat.ptr;
  const size_t m = Amat.rows;
  size_t n = m;

  matrix_identity(Vmat);

  int is_not_diagonal = 0;

  for (int ep = 1; ep <= 20; ep++) {
      double alpha, beta, cos_t, sin_t;

      for (size_t i = 0; i < m; i++) {
          for (size_t j = i + 1; j < m; j++) {
              if (A[i * m + j] != 0.0) {
                  is_not_diagonal = 1;
                  break;
              }
          }
      }

      if (!is_not_diagonal) break;

      for (size_t row = 0; row < m; row++) {
          for (size_t col = row + 1; col < m; col++) {
              __m256d sin_vec, cos_vec;
              // Compute cos_t and sin_t for the rotation

              alpha = 2.0 * sign(A[row * m + row] - A[col * m + col]) * A[row * m + col];
              beta = fabs(A[row * m + row] - A[col * m + col]);
              cos_t = sqrt(0.5 * (1 + beta / sqrt(alpha * alpha + beta * beta)));
              // sin_t = (1 / 2*cos_t) * (alpha / sqrt(alpha*alpha + beta*beta));
              sin_t = sign(alpha) * sqrt(1 - cos_t * cos_t);

              sin_vec = _mm256_set1_pd(sin_t);
              cos_vec = _mm256_set1_pd(cos_t);

              for (size_t i = 0; i < m; i++) {
                  // Compute the eigen values by updating the columns until convergence
                  double A_i_r = A[m * i + row];
                  A[m * i + row] = cos_t * A[m * i + row] + sin_t * A[m * i + col];
                  A[m * i + col] = cos_t * A[m * i + col] - sin_t * A_i_r;
              }

              if (m % 4 != 0)
                  n = m - (m % 4);

              for (size_t i = 0; i < n; i+=4) {
                  __m256d A_row, A_col, A_rcopy, V_row, V_col, V_rcopy;
                  __m256d sin_row, sin_col, cos_row, cos_col;

                  // Compute the eigen values by updating the rows until convergence
                  A_row = _mm256_loadu_pd(A + m * row + i);
                  A_rcopy = A_row;
                  A_col = _mm256_loadu_pd(A + m * col + i);

                  cos_row = _mm256_mul_pd(A_row, cos_vec);
                  sin_col = _mm256_mul_pd(A_col, sin_vec);
                  A_row = _mm256_add_pd(cos_row, sin_col);
                  _mm256_storeu_pd(A + m * row + i, A_row);

                  cos_col = _mm256_mul_pd(A_col, cos_vec);
                  sin_row = _mm256_mul_pd(A_rcopy, sin_vec);
                  A_col = _mm256_sub_pd(cos_col, sin_row);
                  _mm256_storeu_pd(A + m * col + i, A_col);

                  // Compute the eigen vectors similarly by updating the eigen vector matrix
                  V_row = _mm256_loadu_pd(V + m * row + i);
                  V_rcopy = V_row;
                  V_col = _mm256_loadu_pd(V + m * col + i);

                  cos_row = _mm256_mul_pd(V_row, cos_vec);
                  sin_col = _mm256_mul_pd(V_col, sin_vec);
                  V_row = _mm256_add_pd(cos_row, sin_col);
                  _mm256_storeu_pd(V + m * row + i, V_row);

                  cos_col = _mm256_mul_pd(V_col, cos_vec);
                  sin_row = _mm256_mul_pd(V_rcopy, sin_vec);
                  V_col = _mm256_sub_pd(cos_col, sin_row);
                  _mm256_storeu_pd(V + m * col + i, V_col);
              }

              if (m % 4 != 0) {
                  for (size_t i = 0; i < m - n; i++) {
                    double A_r_i = A[m * row + n + i];
                    A[m * row + n + i] = cos_t * A[m * row + n + i] + sin_t * A[m * col + n + i];
                    A[m * col + n + i] = cos_t * A[m * col + n + i] - sin_t * A_r_i;

                    double V_r_i = V[m * row + n + i];
                    V[m * row + n + i] = cos_t * V[m * row + n + i] + sin_t * V[m * col + n + i];
                    V[m * col + n + i] = cos_t * V[m * col + n + i] - sin_t * V_r_i;
                  }
              }
          }
      }
      is_not_diagonal = 0;
  }

  matrix_transpose({V, m, m}, {V, m, m});
}

static inline void mult_block(struct matrix_t Amat, size_t blockA_row, size_t blockA_col, struct matrix_t Bmat,
                              size_t blockB_row, size_t blockB_col, struct matrix_t Cmat, size_t blockC_row,
                              size_t blockC_col, size_t block_size) {
    size_t nA = Amat.rows;
    size_t nB = Bmat.rows;
    size_t nC = Cmat.rows;
    double* A = Amat.ptr;
    double* B = Bmat.ptr;
    double* C = Cmat.ptr;
    size_t Abeg = blockA_row * block_size * nA + blockA_col * block_size;
    size_t Bbeg = blockB_row * block_size * nB + blockB_col * block_size;
    size_t Cbeg = blockC_row * block_size * nC + blockC_col * block_size;
    for (size_t i = 0; i < block_size; ++i) {
        for (size_t j = 0; j < block_size; ++j) {
            C[Cbeg + i * nC + j] = 0.0;
            for (size_t k = 0; k < block_size; ++k) {
                C[Cbeg + i * nC + j] += A[Abeg + i * nA + k] * B[Bbeg + k * nB + j];
            }
        }
    }
}

static inline void add(struct matrix_t Amat, struct matrix_t Bmat, struct matrix_t Cmat) {
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

static inline void copy_block(struct matrix_t Smat, size_t blockS_row, size_t blockS_col,
                      struct matrix_t Dmat, size_t blockD_row, size_t blockD_col, size_t block_size) {
    size_t nS = Smat.rows;
    size_t nD = Dmat.rows;
    double* S = Smat.ptr;
    double* D = Dmat.ptr;
    size_t Sbeg = blockS_row * block_size * nS + blockS_col * block_size;
    size_t Dbeg = blockD_row * block_size * nD + blockD_col * block_size;
    for (size_t i = 0; i < block_size; ++i) {
        for (size_t j = 0; j < block_size; j += 8) {
            D[Dbeg + i * nD + j + 0] = S[Sbeg + i * nS + j + 0];
            D[Dbeg + i * nD + j + 1] = S[Sbeg + i * nS + j + 1];
            D[Dbeg + i * nD + j + 2] = S[Sbeg + i * nS + j + 2];
            D[Dbeg + i * nD + j + 3] = S[Sbeg + i * nS + j + 3];
            D[Dbeg + i * nD + j + 4] = S[Sbeg + i * nS + j + 4];
            D[Dbeg + i * nD + j + 5] = S[Sbeg + i * nS + j + 5];
            D[Dbeg + i * nD + j + 6] = S[Sbeg + i * nS + j + 6];
            D[Dbeg + i * nD + j + 7] = S[Sbeg + i * nS + j + 7];
        }
    }
}

static inline void transpose(struct matrix_t Amat) {
    size_t n = Amat.rows;
    double* A = Amat.ptr;
    for (size_t i = 0; i < n - 1; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double tmp = A[i * n + j];
            A[i * n + j] = A[j * n + i];
            A[j * n + i] = tmp;
        }
    }
}

// perform D = C + AB
static inline void mult_add_block(struct matrix_t Amat, size_t blockA_row, size_t blockA_col, struct matrix_t Bmat,
              size_t blockB_row, size_t blockB_col, struct matrix_t Cmat, size_t blockC_row, size_t blockC_col,
              struct matrix_t Dmat, size_t blockD_row, size_t blockD_col, size_t block_size) {
    size_t nA = Amat.rows;
    size_t nB = Bmat.rows;
    size_t nC = Cmat.rows;
    size_t nD = Dmat.rows;
    double* A = Amat.ptr;
    double* B = Bmat.ptr;
    double* C = Cmat.ptr;
    double* D = Dmat.ptr;
    size_t Abeg = blockA_row * block_size * nA + blockA_col * block_size;
    size_t Bbeg = blockB_row * block_size * nB + blockB_col * block_size;
    size_t Cbeg = blockC_row * block_size * nC + blockC_col * block_size;
    size_t Dbeg = blockD_row * block_size * nD + blockD_col * block_size;
    for (size_t i = 0; i < block_size; ++i) {
        for (size_t j = 0; j < block_size; ++j) {
            D[Dbeg + i * nD + j] = C[Cbeg + i * nC + j];
            for (size_t k = 0; k < block_size; ++k) {
                D[Dbeg + i * nD + j] += A[Abeg + i * nA + k] * B[Bbeg + k * nB + j];
            }
        }
    }
}

// perform C += (A^T)B
// for C_ij, use ith column of A and jth column of B
void inline mult_transpose_block(struct matrix_t Amat, size_t blockA_row, size_t blockA_col, struct matrix_t Bmat,
                          size_t blockB_row, size_t blockB_col, struct matrix_t Cmat, size_t blockC_row,
                          size_t blockC_col, size_t block_size) {
    size_t nA = Amat.rows;
    size_t nB = Bmat.rows;
    size_t nC = Cmat.rows;
    double* A = Amat.ptr;
    double* B = Bmat.ptr;
    double* C = Cmat.ptr;
    size_t Abeg = blockA_row * block_size * nA + blockA_col * block_size;
    size_t Bbeg = blockB_row * block_size * nB + blockB_col * block_size;
    size_t Cbeg = blockC_row * block_size * nC + blockC_col * block_size;
    for (size_t i = 0; i < block_size; ++i) {
        for (size_t j = 0; j < block_size; ++j) {
            C[Cbeg + i * nC + j] = 0.0;
            for (size_t k = 0; k < block_size; ++k) {
                C[Cbeg + i * nC + j] += A[Abeg + k * nA + i] * B[Bbeg + k * nB + j];
            }
        }
    }
}

// perform D = C + (A^T)B
// for D_ij, use ith column of A and jth column of B.
void inline mult_add_transpose_block(struct matrix_t Amat, size_t blockA_row, size_t blockA_col, struct matrix_t Bmat,
                              size_t blockB_row, size_t blockB_col, struct matrix_t Cmat, size_t blockC_row,
                              size_t blockC_col, struct matrix_t Dmat, size_t blockD_row, size_t blockD_col,
                              size_t block_size) {
    size_t nA = Amat.rows;
    size_t nB = Bmat.rows;
    size_t nC = Cmat.rows;
    size_t nD = Dmat.rows;
    double* A = Amat.ptr;
    double* B = Bmat.ptr;
    double* C = Cmat.ptr;
    double* D = Dmat.ptr;
    size_t Abeg = blockA_row * block_size * nA + blockA_col * block_size;
    size_t Bbeg = blockB_row * block_size * nB + blockB_col * block_size;
    size_t Cbeg = blockC_row * block_size * nC + blockC_col * block_size;
    size_t Dbeg = blockD_row * block_size * nD + blockD_col * block_size;
    for (size_t i = 0; i < block_size; ++i) {
        for (size_t j = 0; j < block_size; ++j) {
            D[Dbeg + i * nD + j] = C[Cbeg + i * nC + j];
            for (size_t k = 0; k < block_size; ++k) {
                D[Dbeg + i * nD + j] += A[Abeg + k * nA + i] * B[Bbeg + k * nB + j];
            }
        }
    }
}
