#include <math.h>
#include "evd.hpp"
#include "util.hpp"

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
