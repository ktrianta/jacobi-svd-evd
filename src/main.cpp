#include <iostream>
#include <iomanip>
#include <vector>
#include "evd.hpp"

int main() {
    int n = 5, m = 5;
    double *X = (double*)malloc(sizeof(double)*n*n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            X[n*i + j] = i + j;

    double *e = (double*)malloc(sizeof(double)*n);
    double *Q = (double*)malloc(sizeof(double)*n*n);
    evd_classic(X,n,e,Q);
    std::cout<<e;

}
