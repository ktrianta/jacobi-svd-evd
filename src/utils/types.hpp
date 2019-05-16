#pragma once
#include <cstddef>
#include <vector>
#include "boost/align/aligned_allocator.hpp"

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

/**
 * 32-byte (256 bits) aligned vector.
 */
template <typename T>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, 32>>;
