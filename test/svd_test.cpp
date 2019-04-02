#include "../src/svd.hpp"
#include <math.h>
#include <random>
#include <vector>
#include "../src/types.hpp"
#include "gtest/gtest.h"

TEST(svd, identity_matrix) {
    size_t n_rows = 10, n_cols = 10;
    std::vector<double> X(n_rows * n_cols, 0);
    for (int i = 0; i < n_rows; ++i) {
        X[i * n_cols + i] = 1.0;
    }
    std::vector<double> s(n_rows), U(n_rows * n_cols), V(n_cols * n_cols);
    vector_t svec = {&s[0], n_rows};
    matrix_t Xmat = {&X[0], n_rows, n_cols};
    matrix_t Umat = {&U[0], n_rows, n_cols};
    matrix_t Vmat = {&V[0], n_rows, n_rows};

    svd(Xmat, svec, Umat, Vmat, 100);
    for (int i = 0; i < n_rows; ++i) {
        ASSERT_DOUBLE_EQ(s[i], 1.0);
    }
    for (int i = 0; i < n_rows * n_cols; ++i) {
        ASSERT_DOUBLE_EQ(X[i], U[i]);
        ASSERT_DOUBLE_EQ(X[i], V[i]);
    }
}

TEST(svd, tall_matrix) {
    size_t m = 10, n = 6;
    std::vector<double> X(m * n), s(std::min(m, n)), U(m * n), V(n * n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) X[n * i + j] = i + j;

    matrix_t Xmat = {&X[0], m, n};
    vector_t svec = {&s[0], std::min(m, n)};
    matrix_t Umat = {&U[0], m, n};
    matrix_t Vmat = {&V[0], n, n};
    svd(Xmat, svec, Umat, Vmat, 1000);

    ASSERT_NEAR(s[0], 59.88190459, 1e-8);
    ASSERT_NEAR(s[1], 4.915028256, 1e-8);
    for (int i = 2; i < std::min(m, n); ++i) {
        ASSERT_NEAR(s[i], 0.0, 1e-8);
    }
}

TEST(svd, random_square_matrix) {
    size_t n = 3;
    std::vector<double> X = {
        1.22214449, 0.20082589, -0.75672479, 1.07593684, 0.20025264, 0.38234639, 0.07532444, 1.06219307, 0.10030849,
    };
    std::vector<double> s(n), U(n * n), V(n * n);

    matrix_t Xmat = {&X[0], n, n};
    vector_t svec = {&s[0], n};
    matrix_t Umat = {&U[0], n, n};
    matrix_t Vmat = {&V[0], n, n};
    svd(Xmat, svec, Umat, Vmat, 1000);

    std::vector<double> s_expect = {1.7139574, 1.0490895, 0.74584282};
    std::vector<double> U_expect = {-0.79260517, 0.31581509,  0.5215725,   -0.57332258, -0.09483708,
                                    -0.81382255, -0.20755303, -0.94406926, 0.25623228};
    std::vector<double> VT_expect = {-0.93419518, -0.28848231, 0.20989835, 0.20286303, -0.91350772,
                                     -0.35263329, -0.29347223, 0.28684771, -0.9119169};
    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(s[i], s_expect[i], 1e-7);
    }
    for (int j = 0; j < n; ++j) {
        // equal up to sign
        int sign = (U[j] / U_expect[j] < 0) ? -1 : 1;
        for (int i = 0; i < n; ++i) {
            ASSERT_NEAR(sign * U[i * n + j], U_expect[i * n + j], 1e-7);
            // transpose
            ASSERT_NEAR(sign * V[i * n + j], VT_expect[j * n + i], 1e-7);
        }
    }
}

TEST(svd, evd_eigvalues_crosscheck) {
    size_t n = 5;
    std::vector<double> X = {
        2.000000000000000000e+00 ,6.000000000000000000e+00 ,4.000000000000000000e+00 ,6.000000000000000000e+00 ,4.500000000000000000e+00,
        6.000000000000000000e+00 ,5.000000000000000000e+00 ,8.000000000000000000e+00 ,5.500000000000000000e+00 ,4.500000000000000000e+00,
        4.000000000000000000e+00 ,8.000000000000000000e+00 ,3.000000000000000000e+00 ,6.000000000000000000e+00 ,4.000000000000000000e+00,
        6.000000000000000000e+00 ,5.500000000000000000e+00 ,6.000000000000000000e+00 ,1.000000000000000000e+00 ,2.000000000000000000e+00,
        4.500000000000000000e+00 ,4.500000000000000000e+00 ,4.000000000000000000e+00 ,2.000000000000000000e+00 ,7.000000000000000000e+00};
    std::vector<double> s(n), U(n * n), V(n * n);

    matrix_t Xmat = {&X[0], n, n};
    vector_t svec = {&s[0], n};
    matrix_t Umat = {&U[0], n, n};
    matrix_t Vmat = {&V[0], n, n};
    svd(Xmat, svec, Umat, Vmat, 1000);

    std::vector<double> s_expect = {
        2.415032147975995969e+01,
        5.881509290566617310e+00,
        4.001355036163166012e+00,
        3.262428878677021693e+00,
        1.007738346679503572e+00};

    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(s[i], s_expect[i], 1e-7);
    }
}
