#pragma once
#include <iomanip>
#include <iostream>
#include <string>
#include "../prettyprint/prettyprint.hpp"

static const char DEBUG_HEADER[] = "//";

/**
 * Custom output operator for vector types. With this function the following code becomes possible:
 *
 * struct vector_t vec;
 * std::cout << vec << std::endl;
 */
std::ostream& operator<<(std::ostream& os, const struct vector_t vec);

/**
 * Custom output operator for matrix types. With this function the following code becomes possible:
 *
 * struct matrix_t mat;
 * std::cout << mat << std::endl;
 */
std::ostream& operator<<(std::ostream& os, const struct matrix_t mat);

/**
 * Debugging utility base case.
 */
template <size_t precision = 5, typename T>
void debug(T t) {
    std::cerr << DEBUG_HEADER << std::fixed << std::setprecision(precision) << t << std::endl;
    std::cerr << DEBUG_HEADER << "-------------------" << std::endl;
}

/**
 * Debugging utility function. Pass as many types that implement operator<<, and the function will
 * output each of them to std::cerr, separated by commas.
 *
 * @tparam precision Precision of floating point values to be printed. Use as debug<P>(your types) where
 * P is the number of precisions you want, e.g. 10.
 */
template <size_t precision = 5, typename T, typename... Args>
void debug(T head, Args... rest) {
    std::cerr << DEBUG_HEADER << std::fixed << std::setprecision(precision) << head << ", ";
    debug<precision>(rest...);
}
