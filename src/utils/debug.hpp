#pragma once
#include <iomanip>
#include <iostream>
#include <string>
#include "../prettyprint/prettyprint.hpp"

static const char DEBUG_HEADER[] = "//";

/**
 *
 */
std::ostream& operator<<(std::ostream& os, const struct vector_t vec);
/**
 *
 */
std::ostream& operator<<(std::ostream& os, const struct matrix_t mat);

/**
 *
 */
template <size_t precision = 5, typename T>
void debug(T t) {
    std::cerr << DEBUG_HEADER << std::fixed << std::setprecision(precision) << t << std::endl;
    std::cerr << DEBUG_HEADER << "-------------------" << std::endl;
}

/**
 *
 */
template <size_t precision = 5, typename T, typename... Args>
void debug(T head, Args... rest) {
    std::cerr << DEBUG_HEADER << std::fixed << std::setprecision(precision) << head << ", ";
    debug<precision>(rest...);
}
