#include <algorithm>
#include <iostream>
#include <vector>
#include "evd.hpp"
#include "perf_test.hpp"
#include "types.hpp"

int main() {
    size_t n = 4;
    std::vector<double> A = {7.0, 3.0, 2.0, 1.0, 3.0, 9.0, -2.0, 4.0, 2.0, -2.0, -4.0, 2.0, 1.0, 4.0, 2.0, 3.0};
    std::vector<double> e(n);
    std::vector<double> V(n * n, 0);

    matrix_t Data_matr = {&A[0], n, n};
    vector_t E_vals = {&e[0], n};
    matrix_t E_vecs = {&V[0], n, n};
    std::vector<double> cycles = measure_cycles(evd_classic, Data_matr, E_vecs, E_vals, 100);
    auto median_it = cycles.begin() + cycles.size() / 2;
    std::nth_element(cycles.begin(), median_it, cycles.end());
    std::cout << "evd_classic requires " << *median_it << " cycles (median)" << std::endl;
}
