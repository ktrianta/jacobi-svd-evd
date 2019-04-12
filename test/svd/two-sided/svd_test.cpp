#include "svd.hpp"
#include <math.h>
#include <iostream>
#include <random>
#include <vector>
#include "gtest/gtest.h"
#include "nsvd.hpp"
#include "types.hpp"
#include "../../test_utils.hpp"

TEST(two_sided_svd, usvd) {
    std::vector<double> X;
    struct svd_2x2_params params;

    X = {1.22214449, 0.20082589, -0.75672479, 1.07593684};
    params = nsvd(X[0], X[1], X[2], X[3]);

    ASSERT_NEAR(params.d1, 1.53219955516, 1e-7);
    ASSERT_NEAR(params.d2, 0.95739501104, 1e-7);

    X = {-9.22214449, 8.29982589, -75.672479, -61.07593684};
    params = nsvd(X[0], X[1], X[2], X[3]);

    ASSERT_NEAR(params.d1, 97.26516185, 1e-7);
    ASSERT_NEAR(params.d2, 12.24816256, 1e-7);

    X = {7.43243, 0.0, 0.0, 1.93243};
    params = nsvd(X[0], X[1], X[2], X[3]);

    ASSERT_NEAR(params.d1, 7.43243, 1e-7);
    ASSERT_NEAR(params.d2, 1.93243, 1e-7);
}

TEST(two_sided_svd, identity_matrix) {
    size_t n_rows = 100, n_cols = 100;
    std::vector<double> X(n_rows * n_cols, 0);
    for (size_t i = 0; i < n_rows; ++i) {
        X[i * n_cols + i] = 1.0;
    }
    std::vector<double> B(n_rows * n_cols), U(n_rows * n_cols), V(n_cols * n_cols);
    matrix_t Xmat = {&X[0], n_rows, n_cols};
    matrix_t Bmat = {&B[0], n_rows, n_cols};
    matrix_t Umat = {&U[0], n_rows, n_cols};
    matrix_t Vmat = {&V[0], n_rows, n_rows};

    svd(Xmat, Bmat, Umat, Vmat);

    for (size_t i = 0; i < n_rows * n_cols; ++i) {
        ASSERT_DOUBLE_EQ(X[i], B[i]);
        ASSERT_DOUBLE_EQ(X[i], V[i]);
    }
}

TEST(two_sided_svd, random_square_matrix) {
    size_t n = 3;
    std::vector<double> B(n * n), U(n * n), V(n * n);
    std::vector<double> X = {1.22214449, 0.20082589, -0.75672479, 1.07593684, 0.20025264,
                             0.38234639, 0.07532444, 1.06219307,  0.10030849};
    std::vector<double> s_expect = {1.7139574, 1.0490895, 0.74584282};
    std::vector<double> U_expect = {-0.79260517, 0.31581509,  -0.5215725,  -0.57332258, -0.09483708,
                                    0.81382255,  -0.20755303, -0.94406926, -0.25623228};
    std::vector<double> VT_expect = {-0.93419518, -0.28848231, 0.20989835,  0.20286303, -0.91350772,
                                     -0.35263329, 0.29347223,  -0.28684771, 0.9119169};

    matrix_t Xmat = {&X[0], n, n};
    matrix_t Bmat = {&B[0], n, n};
    matrix_t Umat = {&U[0], n, n};
    matrix_t Vmat = {&V[0], n, n};

    svd(Xmat, Bmat, Umat, Vmat);

    for (size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(B[i * n + i], s_expect[i], 1e-7);
    }
    for (size_t j = 0; j < n; ++j) {
        // equal up to sign
        double sign = (U[j] / U_expect[j] < 0.0) ? -1.0 : 1.0;
        for (size_t i = 0; i < n; ++i) {
            ASSERT_NEAR(sign * U[i * n + j], U_expect[i * n + j], 1e-7);
            // transpose
            ASSERT_NEAR(sign * V[i * n + j], VT_expect[j * n + i], 1e-7);
        }
    }
}

TEST(two_sided_svd, svd_singvalues_crosscheck) {
    size_t n = 5;
    std::vector<double> X = {
        2.000000000000000000e+00, 6.000000000000000000e+00, 4.000000000000000000e+00, 6.000000000000000000e+00,
        4.500000000000000000e+00, 6.000000000000000000e+00, 5.000000000000000000e+00, 8.000000000000000000e+00,
        5.500000000000000000e+00, 4.500000000000000000e+00, 4.000000000000000000e+00, 8.000000000000000000e+00,
        3.000000000000000000e+00, 6.000000000000000000e+00, 4.000000000000000000e+00, 6.000000000000000000e+00,
        5.500000000000000000e+00, 6.000000000000000000e+00, 1.000000000000000000e+00, 2.000000000000000000e+00,
        4.500000000000000000e+00, 4.500000000000000000e+00, 4.000000000000000000e+00, 2.000000000000000000e+00,
        7.000000000000000000e+00};
    std::vector<double> B(n * n), U(n * n), V(n * n);

    matrix_t Xmat = {&X[0], n, n};
    matrix_t Bmat = {&B[0], n, n};
    matrix_t Umat = {&U[0], n, n};
    matrix_t Vmat = {&V[0], n, n};

    svd(Xmat, Bmat, Umat, Vmat);

    std::vector<double> s_expect = {2.415032147975995969e+01, 5.881509290566617310e+00, 4.001355036163166012e+00,
                                    3.262428878677021693e+00, 1.007738346679503572e+00};

    for (size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(B[i * n + i], s_expect[i], 1e-7);
    }
}

TEST(two_sided_svd, random_matrix_big) {
    size_t n = 100;
    std::vector<double> X(n * n);
    std::vector<double> B(n * n);
    std::vector<double> s_expect(n);
    std::vector<double> U(n * n), U_expect(n * n);
    std::vector<double> V(n * n), VT_expect(n * n);

    std::string cmd = "python scripts/svd_testdata.py " + std::to_string(n) + " " + std::to_string(n);
    std::stringstream ss(exec_cmd(cmd.c_str()));
    read_into(ss, &X[0], n * n);
    read_into(ss, &s_expect[0], n);
    read_into(ss, &U_expect[0], n * n);
    read_into(ss, &VT_expect[0], n * n);

    matrix_t Xmat = {&X[0], n, n};
    matrix_t Bmat = {&B[0], n, n};
    matrix_t Umat = {&U[0], n, n};
    matrix_t Vmat = {&V[0], n, n};
    svd(Xmat, Bmat, Umat, Vmat);

    for (size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(B[i * n + i], s_expect[i], 1e-7);
    }
    for (size_t j = 0; j < n; ++j) {
        // equal up to sign
        int sign = (U[j] / U_expect[j] < 0) ? -1 : 1;
        for (size_t i = 0; i < n; ++i) {
            ASSERT_NEAR(sign * U[i * n + j], U_expect[i * n + j], 1e-7);
            // transpose
            ASSERT_NEAR(sign * V[i * n + j], VT_expect[j * n + i], 1e-7);
        }
    }
}