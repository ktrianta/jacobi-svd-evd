#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "perf_test.hpp"
#include "types.hpp"

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::SelfAdjointEigenSolver<MatrixXd> EVD;

size_t base_cost(size_t n) {
    // https://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html#adf397f6bce9f93c4b0139a47e261fc24
    return 9 * n * n * n;
}

size_t run_evd(EVD&& evd, const MatrixXd& m, unsigned int flags) {
    evd.compute(m, flags);
    return base_cost(m.rows());
}

using EVDTolType = decltype(&run_evd);

std::vector<EVDTolType> tol_based_versions = {run_evd};
std::vector<std::string> tol_based_names = {"evd_eigen"};

int main() {
    std::ios_base::sync_with_stdio(false);  // disable synchronization between C and C++ standard streams
    std::cin.tie(NULL);                     // untie cin from cout

    size_t n;
    std::cin >> n;
    std::cerr << "Performance benchmark on array of size " << n << " by " << n << std::endl;

    MatrixXd A(n, n);
    EVD evd;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; j++) {
            std::cin >> A(i, j);
        }
    }

    run_all(tol_based_versions, tol_based_names, evd, A, Eigen::ComputeEigenvectors);
}
