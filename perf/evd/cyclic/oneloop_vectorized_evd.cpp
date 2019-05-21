#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "perf_test.hpp"
#include "evd_cyclic.hpp"
#include "types.hpp"
#include "evd_cost.hpp"

int main() {
    std::ios_base::sync_with_stdio(false);  // disable synchronization between C and C++ standard streams
    std::cin.tie(NULL);                     // untie cin from cout

    size_t n, n_iter = 10;
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

    size_t cost = oneloop_vectorize_cost_evd(n, n_iter);
    bench_func(evd_cyclic_oneloop_vectorize, "evd_cyclic_oneloop_vectorized", cost, Data_matr, Data_matr_copy, E_vecs,
               E_vals, n_iter);
}
