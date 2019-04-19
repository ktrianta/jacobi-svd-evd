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

int main() {
    size_t n = 4;
    std::vector<double> A = {7.0, 3.0, 2.0, 1.0, 3.0, 9.0, -2.0, 4.0, 2.0, -2.0, -4.0, 2.0, 1.0, 4.0, 2.0, 3.0};
    std::vector<double> e(n);
    std::vector<double> V(n * n, 0);
    matrix_t Data_matr = {&A[0], n, n};
    vector_t E_vals = {&e[0], n};
    matrix_t E_vecs = {&V[0], n, n};

    run_all(epoch_based_versions, epoch_based_names, Data_matr, E_vecs, E_vals, 100);
    run_all(tol_based_versions, tol_based_names, Data_matr, E_vecs, E_vals, 1e-8);
}
