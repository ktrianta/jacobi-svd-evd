#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "perf_test.hpp"
#include "svd.hpp"
#include "types.hpp"

using SVDEpochType = decltype(&svd);

double base_cost(int m, int n, int n_iter) {
    double add = n * m + n_iter * (n * n * n + 2.5 * n * n * m + n * n - 2.5 * m * n - 2 * n);
    double mult = n * m + n_iter * (n * n * n + 4.5 * n * n * m + n * n - 4.5 * m * n - 2 * n);
    double div = n * m + n_iter * (1.5 * n * n - 1.5 * n);
    double sqrt = n + n_iter * (n * n - n);

    return add + mult + div + sqrt;
}

using CostFuncType = decltype(&base_cost);

std::vector<SVDEpochType> epoch_based_versions = {
    svd,
};
std::vector<std::string> epoch_based_names = {
    "svd_onesided",
};
std::vector<CostFuncType> epoch_based_cost_fns = {base_cost};

int main() {
    size_t n = 4;
    size_t n_iter = 100;
    std::vector<double> A = {7.0, 3.0, 2.0, 1.0, 3.0, 9.0, -2.0, 4.0, 2.0, -2.0, -4.0, 2.0, 1.0, 4.0, 2.0, 3.0};
    std::vector<double> s(n);
    std::vector<double> U(n * n, 0);
    std::vector<double> V(n * n, 0);
    matrix_t Data_matr = {&A[0], n, n};
    vector_t s_vals = {&s[0], n};
    matrix_t U_mat = {&U[0], n, n};
    matrix_t V_mat = {&V[0], n, n};

    std::vector<double> costs;
    for (const auto& cost_fn : epoch_based_cost_fns) {
        costs.push_back(cost_fn(n, n, n_iter));
    }

    run_all(epoch_based_versions, epoch_based_names, costs, Data_matr, s_vals, U_mat, V_mat, n_iter);
}
