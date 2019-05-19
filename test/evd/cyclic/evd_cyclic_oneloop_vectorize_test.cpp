#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <vector>
#include "../../test_utils.hpp"
#include "evd_cyclic.hpp"
#include "gtest/gtest.h"
#include "types.hpp"

TEST(evd_cyclic_oneloop_vectorize, identity_matrix) {
    size_t n = 12;
    aligned_vector<double> A(n * n, 0);
    aligned_vector<double> A_copy(n * n, 0);
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] = 1.0;
    }
    aligned_vector<double> e(n);
    aligned_vector<double> V(n * n, 0);
    matrix_t Data_matr = {&A[0], n, n};
    matrix_t Data_matr_copy = {&A_copy[0], n, n};
    vector_t E_vals = {&e[0], n};
    matrix_t E_vecs = {&V[0], n, n};

    evd_cyclic_oneloop_vectorize(Data_matr, Data_matr_copy, E_vecs, E_vals, 100);
    for (size_t i = 0; i < n; ++i) {
        ASSERT_DOUBLE_EQ(e[i], 1.0);
    }
}

TEST(evd_cyclic_oneloop_vectorize, random_square_matrix) {
    size_t n = 4;
    aligned_vector<double> A = {7.0, 3.0, 2.0, 1.0, 3.0, 9.0, -2.0, 4.0, 2.0, -2.0, -4.0, 2.0, 1.0, 4.0, 2.0, 3.0};
    aligned_vector<double> A_copy(n * n, 0);
    aligned_vector<double> e(n);
    aligned_vector<double> V(n * n, 0);

    matrix_t Data_matr = {&A[0], n, n};
    matrix_t Data_matr_copy = {&A_copy[0], n, n};
    vector_t E_vals = {&e[0], n};
    matrix_t E_vecs = {&V[0], n, n};

    evd_cyclic_oneloop_vectorize(Data_matr, Data_matr_copy, E_vecs, E_vals, 100);

    aligned_vector<double> e_expect = {12.71986, 5.78305, 2.09733, -5.60024};

    for (size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(e[i], e_expect[i], 1e-5);
    }
}

TEST(evd_cyclic_oneloop_vectorize, random_matrix_big) {
    size_t n = 128;

    aligned_vector<double> A(n * n, 0);
    aligned_vector<double> A_copy(n * n, 0);
    aligned_vector<double> e(n), e_expect(n);
    aligned_vector<double> V(n * n, 0), V_expect(n * n, 0);

    std::string cmd = "python scripts/evd_testdata.py " + std::to_string(n) + " " + std::to_string(n);
    std::stringstream ss(exec_cmd(cmd.c_str()));
    read_into(ss, &A[0], n * n);
    read_into(ss, &e_expect[0], n);
    read_into(ss, &V_expect[0], n * n);

    matrix_t Data_matr = {&A[0], n, n};
    matrix_t Data_matr_copy = {&A_copy[0], n, n};
    vector_t E_vals = {&e[0], n};
    matrix_t E_vecs = {&V[0], n, n};
    evd_cyclic_oneloop_vectorize(Data_matr, Data_matr_copy, E_vecs, E_vals, 100);

    for (size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(e[i], e_expect[i], 1e-7);
    }
    for (size_t j = 0; j < n; ++j) {
        // equal up to sign
        int sign = (V[j] / V_expect[j] < 0) ? -1 : 1;
        for (size_t i = 0; i < n; ++i) {
            ASSERT_NEAR(sign * V[i * n + j], V_expect[i * n + j], 1e-7);
        }
    }
}
