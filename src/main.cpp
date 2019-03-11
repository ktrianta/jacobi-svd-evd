#include <iostream>
#include <iomanip>
#include <vector>
#include "evd.hpp"

int main() {
    int n = 10, m = 10;
    std::vector<double> X(n*m);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            X[n*i + j] = i + j;

    double c, s;
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < m; ++q) {
            sym_schur2(X.data(), n, p, q, &c, &s);
            std::cout << "sym_schur2(p=" << p << ", q=" << q << ") --> c: " << std::fixed << std::setw(8) << c << ", s: " << s << '\n';
        }
    }
    std::cout << std::flush;
}
