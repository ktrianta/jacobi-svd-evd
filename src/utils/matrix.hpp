#pragma once

#include "types.hpp"

/**
 * @brief Initialize an identity matrix
 * @param Pmat Input matrix
 */
void matrix_identity(matrix_t Pmat);

/**
 * @brief Generate the transpose of a matrix
 * @param Pmat Input matrix
 * @param Qmat Output matrix - Transpose of input matrix
 */
void matrix_transpose(matrix_t Pmat, matrix_t Qmat);

/**
 * @brief Multiply two matrices and store the result in another matrix
 * @param Qmat Input left matrix
 * @param Rmat Input right matrix
 * @param Pmat Output matrix
 */
void matrix_mult(matrix_t Pmat, matrix_t Qmat, matrix_t Rmat);
