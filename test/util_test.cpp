#include "gtest/gtest.h"
#include "../src/util.hpp"
#include "../src/types.hpp"
#include <math.h>
#include <vector>
#include <random>

TEST(util, isclose) {
    ASSERT_TRUE(isclose(0, 0)) << "basic sanity check";
    ASSERT_FALSE(isclose(0, 1)) << "not equal at any point";
    ASSERT_TRUE(isclose(1, 1.000001, 1e-6)) << "close for same epsilon";
    ASSERT_FALSE(isclose(1, 1.000001, 1e-7)) << "not close when epsilon is small";
    ASSERT_TRUE(isclose(1e9, 1e9 + 1, 1e-8)) << "relative closeness for very large values";
    ASSERT_TRUE(isclose(1e9, 1e9 - 1, 1e-8)) << "relative closeness for very large values";

    ASSERT_FALSE(isclose(0, 1)) << "negative not equal at any point";
    ASSERT_TRUE(isclose(-1, -1.000001, 1e-6)) << "negative close for same epsilon";
    ASSERT_FALSE(isclose(-1, -1.000001, 1e-7)) << "negative not close when epsilon is small";
    ASSERT_TRUE(isclose(-1e9, -1e9 - 1, 1e-8)) << "negative relative closeness for very large values";
    ASSERT_TRUE(isclose(-1e9, -1e9 + 1, 1e-8)) << "negative relative closeness for very large values";
}

TEST(util, sym_jacobi_coeffs) {
    double c, s;
    double x_ii, x_ij, x_jj;
    double t;

    x_ii = x_ij = x_jj = 0;
    sym_jacobi_coeffs(x_ii, x_ij, x_jj, &c, &s);
    ASSERT_DOUBLE_EQ(c, 1.0) << "x_ij = 0 case";
    ASSERT_DOUBLE_EQ(s, 0.0) << "x_ij = 0 case";

    x_ii = x_jj = 0;
    x_ij = 124;
    sym_jacobi_coeffs(x_ii, x_ij, x_jj, &c, &s);
    ASSERT_DOUBLE_EQ(c, 1.0/sqrt(2)) << "x_ii = x_jj = 0, x_ij != 0 case";
    ASSERT_DOUBLE_EQ(s, 1.0/sqrt(2)) << "x_ii = x_jj = 0, x_ij != 0 case";

    x_ij = 0.25;
    x_ii = 0, x_jj = sqrt(2);
    sym_jacobi_coeffs(x_ii, x_ij, x_jj, &c, &s);
    t = 1/(2*sqrt(2) + 3);
    ASSERT_DOUBLE_EQ(c, 1.0/sqrt(1 + t*t)) << "x_ii = 0, x_jj = sqrt(2), x_ij = 1/4 case";
    ASSERT_DOUBLE_EQ(s, t/sqrt(1 + t*t)) << "x_ii = 0, x_jj = sqrt(2), x_ij = 1/4 case";

    x_ij = 0.25;
    x_ii = sqrt(2), x_jj = 0;
    sym_jacobi_coeffs(x_ii, x_ij, x_jj, &c, &s);
    t = -1/(2*sqrt(2) + 3);
    ASSERT_DOUBLE_EQ(c, 1.0/sqrt(1 + t*t)) << "x_ii = 0, x_jj = sqrt(2), x_ij = 1/4 case";
    ASSERT_DOUBLE_EQ(s, t/sqrt(1 + t*t)) << "x_ii = 0, x_jj = sqrt(2), x_ij = 1/4 case";
}

TEST(util, reorder_decomposition) {
    size_t n_cols = 10;
    size_t n_rows[3] = {15, 5, 1};

    // define ordered columns. Results of the shuffled inputs must be equal
    // to the original ones.
    std::vector<double> s(n_cols);
    std::iota(s.begin(), s.end(), 0);

    std::vector<double> orig_mats[3] = {
        std::vector<double>(n_rows[0]*n_cols),
        std::vector<double>(n_rows[1]*n_cols),
        std::vector<double>(n_rows[2]*n_cols)
    };
    for (int i = 0; i < n_cols; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < n_rows[j]; ++k) {
                orig_mats[j][k*n_cols + i] = i;
            }
        }
    }

    // copy to modify. Randomly shuffle values and corresponding columns.
    std::vector<double> s_copy(s);
    std::random_shuffle(s_copy.begin(), s_copy.end());
    std::vector<double> copy_mats[3] = {
        std::vector<double>(orig_mats[0]),
        std::vector<double>(orig_mats[1]),
        std::vector<double>(orig_mats[2])
    };
    for (int i = 0; i < n_cols; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < n_rows[j]; ++k) {
                copy_mats[j][k*n_cols + i] = orig_mats[j][k*n_cols + s_copy[i]];
            }
        }
    }

    // run the function
    vector_t svec = {&s_copy[0], s_copy.size()};
    matrix_t matrices[3] = {
        {&copy_mats[0][0], n_rows[0], n_cols},
        {&copy_mats[1][0], n_rows[1], n_cols},
        {&copy_mats[2][0], n_rows[2], n_cols}
    };
    reorder_decomposition(svec, matrices, 3, less);

    // assert that we recover the original ordering in every cell.
    for (int i = 0; i < n_cols; ++i) {
        ASSERT_DOUBLE_EQ(s[i], s_copy[i]) << "singular values are ordered";
    }
    for (int i = 0; i < n_cols; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < n_rows[j]; ++k) {
                ASSERT_DOUBLE_EQ(orig_mats[j][k*n_cols + i], copy_mats[j][k*n_cols + i])
                    << "columns are in the same ordering";
            }
        }
    }
}
