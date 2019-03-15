#include <math.h>
#include "evd.hpp"
#include "util.hpp"
#include <iostream>

void sym_schur2(const double* const X, int n, int p, int q, double* c, double* s) {
    if (not isclose(X[n*p + q], 0)) {
        double tau, t, out_c;
        tau = (X[n*q + q] - X[n*p + p])/(2*X[n*p + q]);
        if (tau >= 0) {
            t = 1.0/(tau + sqrt(1 + tau*tau));
        } else {
            t = -1.0/(-tau + sqrt(1 + tau*tau));
        }
        out_c = 1.0/sqrt(1 + t*t);
        *c = out_c;
        *s = t*out_c;
    } else {
        *c = 1.0;
        *s = 0.0;
    }
}

void evd_classic(const double* const X, int n, double* e, double* Q) {

    double offX2;
    //Definition of off, however we use the square
    for(int i = 0; i < n; ++i){
        for(int j = i+1; j<n; ++j){
            double a_ij = X[n*i+j];
            offX2+=2*a_ij*a_ij;
        }
    }

    double abs_a, eps = offX2;
    //Frebonius norm
    for(int i = 0; i < n; ++i){
        eps += X[n*i+i]*X[n*i+i];
    }
    //tol*frebonius norm
    double tol = 0.001;
    eps = tol*tol*eps;

    int p, q;
    double c, s;
    double *A = (double*)malloc(sizeof(double)*n*n);

    for(int i = 0; i < n; ++i){
        for(int j = i+1; j<n; ++j){
            A[n*i+j] = X[n*i+j];
        }
    }

    for(int i = 0; i < n; ++i){
        Q[n*i+i] = 1.0;
    }

    std::cout << eps << "\n";
    while(offX2 > eps){
        abs_a = 0.0;
        for(int i = 0; i < n; ++i){
            for(int j = i+1; j<n; ++j){
                double abs_ij = abs(A[i*n+j]);
                if(abs_ij > abs_a){
                    abs_a = abs_ij;
                    p = i;
                    q = j;
                }
            }
        }
        sym_schur2(A,n,p,q,&c,&s);
        for(int i = 0; i < n; ++i){

            Q[n*i+p] = c*Q[n*i+p]-s*Q[n*i+q];
            Q[n*i+q] = s*Q[n*i+p]+c*Q[n*i+q];

            A[n*i+p] = c * A[n*i+p] - s * A[n*i+q];
            A[n*i+q] = s * A[n*i+p] + c * A[n*i+q];
            A[n*p+i] = c * A[n*i+p] - s * A[n*i+q];
            A[n*q+i] = s * A[n*i+p] + c * A[n*i+q];
        }
        offX2 = 0.0;
        for(int i = 0; i < n; ++i){
            for(int j = i+1; j<n; ++j){
                double a_ij = A[n*i+j];
                offX2+=2*a_ij*a_ij;
            }
        }
        std::cout << offX2 << "\n";
    }

    for(int i = 0; i < n; ++i){
        e[i]=A[n*i+i];
    }
}
