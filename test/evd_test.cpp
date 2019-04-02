#include "../src/evd.hpp"
#include <math.h>
#include <vector>
#include "../src/types.hpp"
#include "gtest/gtest.h"

TEST(evd, identity_matrix) {
    size_t n = 10;
    std::vector<double> A(n * n, 0);
    for (int i = 0; i < n; ++i) {
        A[i * n + i] = 1.0;
    }
    std::vector<double> e(n);
    std::vector<double> V(n * n, 0);
    matrix_t Data_matr = {&A[0], n, n};
    vector_t E_vals = {&e[0], n};
    matrix_t E_vecs = {&V[0], n, n};

    evd_classic(Data_matr, E_vecs, E_vals, 100);
    for (int i = 0; i < n; ++i) {
        ASSERT_DOUBLE_EQ(e[i], 1.0);
    }
}

TEST(evd, random_square_matrix) {
    size_t n = 4;
    std::vector<double> A = {7.0, 3.0, 2.0, 1.0, 3.0, 9.0, -2.0, 4.0, 2.0, -2.0, -4.0, 2.0, 1.0, 4.0, 2.0, 3.0};
    std::vector<double> e(n);
    std::vector<double> V(n * n, 0);

    matrix_t Data_matr = {&A[0], n, n};
    vector_t E_vals = {&e[0], n};
    matrix_t E_vecs = {&V[0], n, n};

    evd_classic(Data_matr, E_vecs, E_vals, 100);

    std::vector<double> e_expect = {12.71986, 5.78305, 2.09733, -5.60024};

    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(e[i], e_expect[i], 1e-2);
    }
}
