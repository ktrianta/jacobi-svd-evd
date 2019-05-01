#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "evd_cyclic.hpp"
#include "perf_test.hpp"
#include "types.hpp"

using EVDEpochType = decltype(&evd_cyclic);
using EVDTolType = decltype(&evd_cyclic_tol);

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

double tol_cost(size_t n, size_t n_iter) {
    double n_elements = n * (n - 1) / 2;
    double adds = (4 + 6 * n) * n_elements + n_elements * 2;
    double muls = (4 + 12 * n) * n_elements + n_elements * 2;
    double divs = 3 * n_elements;
    double sqrt = 2 * n_elements;

    return n_iter * (adds + muls + divs + sqrt);
}
double base_cost_evd(size_t n, size_t n_iter) {
    double add = n_iter * (5 + 6 * n);
    double mult = n_iter * (7 + 12 * n);
    double div = n_iter;
    double sqrt = n_iter * 3;

    return add + mult + div + sqrt;
}

using CostFuncType = decltype(&tol_cost);

std::vector<CostFuncType> tol_based_cost_fns = {tol_cost};

int main() {
    size_t n;

    std::ios_base::sync_with_stdio(false);  // disable synchronization between C and C++ standard streams
    std::cin.tie(NULL);                     // untie cin from cout

    std::cin >> n;
    std::cout << "Performance benchmark on array of size " << n << " by " << n << std::endl;

    std::vector<double> A(n * n);
    std::vector<double> A_copy(n * n);
    std::vector<double> e(n);
    std::vector<double> V(n * n, 0);
    matrix_t Data_matr = {&A[0], n, n};
    matrix_t Data_matr_copy = {&A_copy[0], n, n};
    vector_t E_vals = {&e[0], n};
    matrix_t E_vecs = {&V[0], n, n};

    for (size_t i = 0; i < n * n; ++i) {
        std::cin >> A[i];
    }

    size_t n_iter = evd_cyclic_tol(Data_matr, Data_matr_copy, E_vecs, E_vals, 1e-8);
    std::cout << n_iter;
    std::vector<double> costs(1, 10000);
    std::vector<double> costs_tol;

    for (const auto& cost_fn : tol_based_cost_fns) {
        costs_tol.push_back(cost_fn(n, n_iter));
    }

    run_all(epoch_based_versions, epoch_based_names, costs, Data_matr, Data_matr_copy, E_vecs, E_vals, 100);
    run_all(tol_based_versions, tol_based_names, costs_tol, Data_matr, Data_matr_copy, E_vecs, E_vals, 1e-8);
}
