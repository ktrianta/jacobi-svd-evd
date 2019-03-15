#include <iostream>
#include <iomanip>
#include <vector>
#include "svd.hpp"

int main() {
    int n = 10, m = 10;
    std::vector<double> X(n*m), s(std::min(m, n)), U(n*m), V(n*n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            X[n*i + j] = i + j;

    svd(X.data(), m, n, &s[0], &U[0], &V[0], 100);

    std::cout << "Original matrix (X)\n";
    std::cout << "-------------------\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setw(7) << std::setprecision(4) << X[n*i + j] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "Singular values (S)\n";
    std::cout << "-------------------\n";
    for (int i = 0; i < std::min(m, n); ++i) {
        std::cout << s[i] << ' ';
    }
    std::cout << '\n';

    std::cout << "Left singular vectors (U)\n";
    std::cout << "-------------------------\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setw(7) << std::setprecision(4) << U[n*i + j] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "Right singular vectors (V)\n";
    std::cout << "---------------------------\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setw(7) << std::setprecision(4) << V[n*i + j] << ' ';
        }
        std::cout << '\n';
    }
}
