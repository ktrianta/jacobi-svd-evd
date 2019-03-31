#pragma once
#include<cstddef>

/**
 * A simple vector type that holds the pointer to an already allocated array
 * together with its size.
 */
struct vector_t {
    double* ptr;
    size_t len;
};

/**
 * A simple matrix type that holds the pointer to an already allocated
 * array together with its number of rows and columns.
 */
struct matrix_t {
    double* ptr;
    size_t rows;
    size_t cols;
};
