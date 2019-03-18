#include "gtest/gtest.h"
#include "../src/evd.hpp"
#include "../src/types.hpp"
#include <math.h>
#include <vector>

TEST(evd, identity_matrix) {
    int n;
    std::vector<double> A(n * n, 0);
    for (int i = 0; i < n; ++i) {
        A[i*n_cols + i] = 1.0;
    }
    std::vector<double> e(n);
    vector_t E_vals = {&e[0], n};
    matrix_t Data_matr = {&A[0], n, n};

    evd_classic(Data_matr, E_vals, 100);
    for (int i = 0; i < n; ++i) {
        ASSERT_DOUBLE_EQ(e[i], 1.0);
    }
}

TEST(evd, random_square_matrix) {
    int n = 4;
    std::vector<double> A = {
       7.0,  3.0,  2.0,  1.0,
       3.0,  9.0,  -2.0,  4.0,
       2.0,  -2.0,  -4.0,  2.0,
	   1.0,  4.0,  2.0,  3.0
    };
    std::vector<double> e(n);

    matrix_t Data_matr = {&X[0], n, n};
    vector_t E_vals = {&s[0], n};
    evd_classic(Data_matr, E_vals, 100);

    std::vector<double> e_expect = {
        5.783165,  12.718053, -5.600226, 2.099008
    };

    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(e[i], e_expect[i], 1e-7);
    }
}