#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "perf_test.hpp"
#include "svd.hpp"
#include "types.hpp"

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::JacobiSVD<MatrixXd> SVD;
size_t base_cost(size_t n, size_t n_iter) {
    // Ops in real_2x2_jacobi svd
    double svd_2x2_wild = 6;

    // Ops in m.applyOnTheLeft(..), part of real_2x2_jacobi svd
    double rot_2x2_add = 2;
    double rot_2x2_mul = 4;

    // Ops in makeJacobi(), part of real_2x2_jacobi svd
    double make_add = 5;
    double make_mul = 4;
    double make_div = 4;
    double make_sqrt = 2;
    double make_abs = 5;

    // Ops in rot1 * j_right->transpose(), part of real_2x2_jacobi svd
    double rot_mul_add = 2;
    double rot_mul_mul = 4;

    double svd_2x2 = svd_2x2_wild + rot_2x2_add + rot_2x2_mul + rot_mul_add + rot_mul_mul + make_add + make_mul +
                     make_div + make_sqrt + make_abs;

    // Ops in m_workMatrix.applyOnTheLeft(..)
    // Same for m_workMatrix.applyOnTheRight(..), m_matrixU.applyOnTheRight(..), m_matrixV.applyOnTheRight(..)
    double rot_add = 2 * n;
    double rot_mul = 4 * n;

    double loops = (n * n - n) / 2;
    double flops = n_iter * loops * (svd_2x2 + 4 * (rot_add + rot_mul));

    return flops;
}

size_t run_jacobi(SVD&& svd, MatrixXd&& m, unsigned int flags) {
    svd.compute(m, flags);
    size_t n_iter = svd.getSweeps();
    return base_cost(m.rows(),n_iter);
}

using SVDTolType = decltype(&run_jacobi);

std::vector<SVDTolType> tol_based_versions = {
    run_jacobi,
};
std::vector<std::string> tol_based_names = {
    "svd_two_sided_eigen",
};

int main() {
    size_t n;

    std::ios_base::sync_with_stdio(false);  // disable synchronization between C and C++ standard streams
    std::cin.tie(NULL);                     // untie cin from cout

    std::cin >> n >> n;
    std::cerr << "Performance benchmark on array of size " << n << " by " << n << std::endl;

    MatrixXd Ap(n, n);
    SVD svd;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; j++) {
            std::cin >> Ap(i, j);
        }
    }

    run_all(tol_based_versions, tol_based_names, svd, Ap, Eigen::ComputeFullU | Eigen::ComputeFullV);
}
