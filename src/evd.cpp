#include <math.h>
#include "evd.hpp"
#include "util.hpp"
#include "types.hpp"
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

void evd_classic_tol(struct matrix_t Xmat, struct vector_t evec, struct matrix_t Qmat, double tol) {
    const int n = Xmat.cols;
    double* X = Xmat.ptr;
    double* e = evec.ptr;
    double* Q = Qmat.ptr;
    //A=QtXQ
    double *A = (double*)malloc(sizeof(double)*n*n);

    double offA=0, abs_a, eps, c, s;
    int p, q;

    for(int i = 0; i < n; ++i){
        for(int j = 0; j<n; ++j){
            A[n*i+j] = X[n*i+j];
        }
    }
    for(int i = 0; i < n; ++i){
        Q[n*i+i] = 1.0;
    }
    for(int i = 0; i < n; ++i){
        for(int j = i+1; j<n; ++j){
            double a_ij = A[n*i+j];
            offA+=2*a_ij*a_ij;
        }
    }
    for(int i = 0; i < n; ++i){
        for(int j = i; j<n; ++j){
            double a_ij = A[n*i+j];
            if(i == j){
                eps+=a_ij*a_ij;
            } else {
                eps+=2*a_ij*a_ij;
            }
        }
    }
    eps = tol*tol*eps;

    while(offA > eps){

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
        double A_ip, A_iq;
        for(int i = 0; i < n; ++i){
            Q[n*i+p] = c*Q[n*i+p]-s*Q[n*i+q];
            Q[n*i+q] = s*Q[n*i+p]+c*Q[n*i+q];

            A_iq = A[n*i+p];
            A_iq = A[n*i+q];

            A[n*i+p] = c * A_ip - s * A_iq;
            A[n*i+q] = s * A_ip + c * A_iq;

        }
        for(int i = 0; i < n; ++i){
            double A_ip = A[n*p+i];
            double A_iq = A[n*q+i];

            A[n*p+i] = c * A_ip - s * A_iq;
            A[n*q+i] = s * A_ip + c * A_iq;
        }

        offA = 0;
        for(int i = 0; i < n; ++i){
            for(int j = i; j < n; ++j){
                double a_ij= A[n*i+j];
                offA+=2*a_ij*a_ij;
            }
        }
    }
    for(int i = 0; i < n; ++i){
        e[i]=A[n*i+i];
    }
    reorder_decomposition(evec, &Qmat, 1, greater);

}
