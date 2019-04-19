#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "perf_test.hpp"
#include "svd.hpp"
#include "types.hpp"

using SVDEpochType = decltype(&svd);

std::vector<SVDEpochType> epoch_based_versions = {
    svd,
};
std::vector<std::string> epoch_based_names = {
    "svd_onesided",
};

int main() {
    size_t n = 4;
    std::vector<double> A = {7.0, 3.0, 2.0, 1.0, 3.0, 9.0, -2.0, 4.0, 2.0, -2.0, -4.0, 2.0, 1.0, 4.0, 2.0, 3.0};
    std::vector<double> s(n);
    std::vector<double> U(n * n, 0);
    std::vector<double> V(n * n, 0);
    matrix_t Data_matr = {&A[0], n, n};
    vector_t s_vals = {&s[0], n};
    matrix_t U_mat = {&U[0], n, n};
    matrix_t V_mat = {&V[0], n, n};

    run_all(epoch_based_versions, epoch_based_names, Data_matr, s_vals, U_mat, V_mat, 100);
}
