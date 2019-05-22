#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "evd_cyclic.hpp"
#include "perf_test.hpp"
#include "types.hpp"

using EVDEpochType = decltype(&evd_cyclic_vectorize);
using EVDEpochType = decltype(&evd_cyclic_oneloop_vectorize);
using EVDEpochType = decltype(&evd_cyclic_unroll_outer_vectorize);

std::vector<EVDEpochType> epoch_based_versions = {evd_cyclic_vectorize, evd_cyclic_oneloop_vectorize,
                                                  evd_cyclic_unroll_outer_vectorize};
std::vector<std::string> epoch_based_names = {"evd_cyclic_vectorize", "evd_cyclic_oneloop_vectorize",
                                              "evd_cyclic_unroll_outer_vectorize"};

double base_cost_evd(size_t n, size_t n_iter) {
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

double one_loop_cost(size_t n, size_t n_iter) {
    // number of upper triangular elements
    double n_elements = n * (n - 1) / 2;
    double adds = (8 + 4 * n) * n_elements;
    double muls = (12 + 8 * n) * n_elements;
    double divs = 3 * n_elements;
    double sqrt = 2 * n_elements;

    return n_iter * (adds + muls + divs + sqrt);
}

double tol_cost(size_t n, size_t n_iter) {
    // number of upper triangular elements
    double n_elements = n * (n - 1) / 2;
    double adds = (4 + 6 * n) * n_elements + n_elements * 2;
    double muls = (4 + 12 * n) * n_elements + n_elements * 2;
    double divs = 3 * n_elements;
    double sqrt = 2 * n_elements;

    return n_iter * (adds + muls + divs + sqrt);
}

using CostFuncType = decltype(&one_loop_cost);
using CostFuncType = decltype(&base_cost_evd);
using CostFuncType = decltype(&tol_cost);

std::vector<CostFuncType> tol_based_cost_fns = {tol_cost};
std::vector<CostFuncType> evdcyclic_epoch_based_cost_fns = {base_cost_evd, one_loop_cost, one_loop_cost};

int main() {
    size_t n;
    size_t n_iter = 100;
    double tol = 1e-8;

    std::ios_base::sync_with_stdio(false);  // disable synchronization between C and C++ standard streams
    std::cin.tie(NULL);                     // untie cin from cout

    std::cin >> n;
    std::cerr << "Performance benchmark on array of size " << n << " by " << n << std::endl;

    aligned_vector<double> A(n * n);
    aligned_vector<double> A_copy(n * n);
    aligned_vector<double> e(n);
    aligned_vector<double> V(n * n, 0);
    matrix_t Data_matr = {&A[0], n, n};
    matrix_t Data_matr_copy = {&A_copy[0], n, n};
    vector_t E_vals = {&e[0], n};
    matrix_t E_vecs = {&V[0], n, n};

    for (size_t i = 0; i < n * n; ++i) {
        std::cin >> A[i];
    }

    std::vector<double> costs;
    for (const auto& cost_fn : evdcyclic_epoch_based_cost_fns) {
        costs.push_back(cost_fn(n, n_iter));
    }

    size_t n_iter_tol = evd_cyclic_tol(Data_matr, Data_matr_copy, E_vecs, E_vals, tol);
    std::vector<double> costs_tol;

    for (const auto& cost_fn : tol_based_cost_fns) {
        costs_tol.push_back(cost_fn(n, n_iter_tol));
    }

    run_all(epoch_based_versions, epoch_based_names, costs, Data_matr, Data_matr_copy, E_vecs, E_vals, n_iter);
}
