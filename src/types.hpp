#pragma once

/**
 * A simple vector type that holds the pointer to an already allocated array
 * together with its size.
 */
struct vector_t {
    double* ptr;
    int len;
};

/**
 * A simple matrix type that holds the pointer to an already allocated
 * array together with its number of rows and columns.
 */
struct matrix_t {
    double* ptr;
    int rows;
    int cols;
};
