#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "evd_cyclic.hpp"
#include "perf_test.hpp"
#include "types.hpp"

using EVDEpochType = decltype(&evd_cyclic);
using EVDTolType = decltype(&evd_cyclic_tol);

double base_cost_evd(int n, int n_iter) {
    // The loop_multiplier counts the total number of values
    // in the upper triangular matrix each corresponding to
    // an iteration
    int loop_multiplier = n * (n - 1) * 0.5;

    double add = n_iter * (5 + 6 * n);
    double mult = n_iter * (7 + 12 * n);
    double div = n_iter;
    double sqrt = n_iter * 3;

    return (add + mult + div + sqrt) * loop_multiplier;
}

using CostFuncType = decltype(&base_cost_evd);

std::vector<EVDEpochType> epoch_based_versions = {
    evd_cyclic,
};
std::vector<std::string> epoch_based_names = {
    "evd_cyclic",
};

std::vector<EVDTolType> tol_based_versions = {
    evd_cyclic_tol,
};
std::vector<std::string> tol_based_names = {
    "evd_cyclic_tol",
};

std::vector<CostFuncType> evdcyclic_epoch_based_cost_fns = {base_cost_evd};

int main() {
    size_t n = 4;
    size_t n_iter = 20;
    std::vector<double> A = {7.0, 3.0, 2.0, 1.0, 3.0, 9.0, -2.0, 4.0, 2.0, -2.0, -4.0, 2.0, 1.0, 4.0, 2.0, 3.0};
    std::vector<double> A_copy(n * n);
    std::vector<double> e(n);
    std::vector<double> V(n * n, 0);
    matrix_t Data_matr = {&A[0], n, n};
    matrix_t Data_matr_copy = {&A_copy[0], n, n};
    vector_t E_vals = {&e[0], n};
    matrix_t E_vecs = {&V[0], n, n};

    std::vector<double> costs;
    for (const auto& cost_fn : evdcyclic_epoch_based_cost_fns) {
        costs.push_back(cost_fn(n, n_iter));
    }

    run_all(epoch_based_versions, epoch_based_names, costs, Data_matr, Data_matr_copy, E_vecs, E_vals, 100);
    run_all(tol_based_versions, tol_based_names, costs, Data_matr, Data_matr_copy, E_vecs, E_vals, 1e-8);
}
