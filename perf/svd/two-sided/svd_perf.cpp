#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "perf_test.hpp"
#include "svd.hpp"
#include "types.hpp"

using SVDTolType = decltype(&svd);

double base_cost(size_t n, size_t n_iter) {
    double rot_add = 8 * n;
    double rot_mul = 20 * n;

    double svd_add = 26;
    double svd_mul = 26;
    double svd_div = 5;
    double svd_sqrt = 3;

    double loops = (n * n - n) / 2;

    double frobenius_add = n * n;
    double frobenius_mul = n * n;

    double add = n_iter * (loops * (rot_add + svd_add) + frobenius_add);
    double mult = n_iter * (loops * (rot_mul + svd_mul) + frobenius_mul);
    double div = n_iter * loops * svd_div;
    double sqrt = n_iter * loops * svd_sqrt;

    return add + mult + div + sqrt;
}

using CostFuncType = decltype(&base_cost);

std::vector<SVDTolType> tol_based_versions = {
    svd,
};
std::vector<std::string> tol_based_names = {
    "svd_two_sided",
};
std::vector<CostFuncType> tol_based_cost_fns = {base_cost};

int main() {
    size_t n;

    std::ios_base::sync_with_stdio(false);  // disable synchronization between C and C++ standard streams
    std::cin.tie(NULL);  // untie cin from cout

    std::cin >> n >> n;
    std::cout << "Performance benchmark on array of size " << n << " by " << n << std::endl;

    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    std::vector<double> U(n * n, 0);
    std::vector<double> V(n * n, 0);
    matrix_t Data_matr = {&A[0], n, n};
    matrix_t B_mat = {&B[0], n, n};
    matrix_t U_mat = {&U[0], n, n};
    matrix_t V_mat = {&V[0], n, n};

    for (size_t i = 0; i < n * n; ++i) {
        std::cin >> A[i];
    }

    size_t n_iter = svd(Data_matr, B_mat, U_mat, V_mat);

    std::vector<double> costs;
    for (const auto& cost_fn : tol_based_cost_fns) {
        costs.push_back(cost_fn(n, n_iter));
    }

    run_all(tol_based_versions, tol_based_names, costs, Data_matr, B_mat, U_mat, V_mat);
}
