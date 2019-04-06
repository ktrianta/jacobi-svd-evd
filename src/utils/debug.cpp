#include "debug.hpp"
#include "types.hpp"

std::ostream& operator<<(std::ostream& os, const struct vector_t vec) {
    os << DEBUG_HEADER << "[\n";
    for (size_t i = 0; i < vec.len; ++i) {
        os << DEBUG_HEADER << '\t' << vec.ptr[i] << ",\n";
    }
    os << ']' << std::endl;
    return os;
}

std::ostream& operator<<(std::ostream& os, const struct matrix_t mat) {
    os << DEBUG_HEADER << "[\n";
    for (size_t i = 0; i < mat.rows; ++i) {
        os << DEBUG_HEADER << "\t[";
        if (mat.cols > 0) {
            os << mat.ptr[0];
        }
        for (size_t j = 1; j < mat.cols; ++j) {
            os << ", " << mat.ptr[j];
        }
        os << "]\n";
    }
    os << ']' << std::endl;
    return os;
}
