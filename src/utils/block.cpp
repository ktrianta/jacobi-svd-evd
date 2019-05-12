#include "block.hpp"

void copy_block(struct matrix_t Smat, size_t blockS_row, size_t blockS_col, struct matrix_t Dmat, size_t blockD_row,
                size_t blockD_col, size_t block_size) {
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

// perform C = AB
void mult_block(struct matrix_t Amat, size_t blockA_row, size_t blockA_col, struct matrix_t Bmat, size_t blockB_row,
                size_t blockB_col, struct matrix_t Cmat, size_t blockC_row, size_t blockC_col, size_t block_size) {
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

// perform D = C + AB
void mult_add_block(struct matrix_t Amat, size_t blockA_row, size_t blockA_col, struct matrix_t Bmat, size_t blockB_row,
                    size_t blockB_col, struct matrix_t Cmat, size_t blockC_row, size_t blockC_col, struct matrix_t Dmat,
                    size_t blockD_row, size_t blockD_col, size_t block_size) {
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

// perform C = (A^T)B
// for C_ij, use ith column of A and jth column of B
void mult_transpose_block(struct matrix_t Amat, size_t blockA_row, size_t blockA_col, struct matrix_t Bmat,
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
void mult_add_transpose_block(struct matrix_t Amat, size_t blockA_row, size_t blockA_col, struct matrix_t Bmat,
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

// perform C = A(B^T)
void mult_transpose_block_right(struct matrix_t Amat, size_t blockA_row, size_t blockA_col, struct matrix_t Bmat,
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
                C[Cbeg + i * nC + j] += A[Abeg + i * nA + k] * B[Bbeg + j * nB + k];
            }
        }
    }
}
// perform C = C + A(B^T)
void mult_add_transpose_block_right(struct matrix_t Amat, size_t blockA_row, size_t blockA_col, struct matrix_t Bmat,
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
                D[Dbeg + i * nD + j] += A[Abeg + i * nA + k] * B[Bbeg + j * nB + k];
            }
        }
    }
}
