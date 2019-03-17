#include <iostream>
#include <iomanip>
#include <vector>
#include "svd.hpp"
#include "types.hpp"

int main() {
    int m = 10, n = 6;
    std::vector<double> X(m*n), s(std::min(m, n)), U(m*n), V(n*n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            X[n*i + j] = i + j;

    matrix_t Xmat = {&X[0], m, n};
    vector_t svec = {&s[0], std::min(m, n)};
    matrix_t Umat = {&U[0], m, n};
    matrix_t Vmat = {&V[0], n, n};
    svd(Xmat, svec, Umat, Vmat, 100);

    std::cout << "Original matrix (X)\n";
    std::cout << "-------------------\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setw(13) << std::setprecision(10) << X[n*i + j] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "Singular values (S)\n";
    std::cout << "-------------------\n";
    for (int i = 0; i < std::min(m, n); ++i) {
        std::cout << std::fixed << std::setw(13) << std::setprecision(10) << s[i] << ' ';
    }
    std::cout << '\n';

    std::cout << "Left singular vectors (U)\n";
    std::cout << "-------------------------\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setw(13) << std::setprecision(10) << U[n*i + j] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "Right singular vectors (V)\n";
    std::cout << "---------------------------\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setw(13) << std::setprecision(10) << V[n*i + j] << ' ';
        }
        std::cout << '\n';
    }
}
