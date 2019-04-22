#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "evd_classic.hpp"
#include "perf_test.hpp"
#include "types.hpp"

using EVDEpochType = decltype(&evd_classic);
using EVDTolType = decltype(&evd_classic_tol);

std::vector<EVDEpochType> epoch_based_versions = {
    evd_classic,
};
std::vector<std::string> epoch_based_names = {
    "evd_classic",
};

std::vector<EVDTolType> tol_based_versions = {
    evd_classic_tol,
};
std::vector<std::string> tol_based_names = {
    "evd_classic_tol",
};

int main() {
    size_t n;

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

    std::vector<double> costs(1, 10000);

    run_all(epoch_based_versions, epoch_based_names, costs, Data_matr, Data_matr_copy, E_vecs, E_vals, 100);
    run_all(tol_based_versions, tol_based_names, costs, Data_matr, Data_matr_copy, E_vecs, E_vals, 1e-8);
}
