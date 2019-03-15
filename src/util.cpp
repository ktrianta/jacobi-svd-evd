#include <math.h>
#include "util.hpp"

bool isclose(double x, double y, double eps) {
    return fabs(x - y) < eps * fabs(x + y);
}

void sym_jacobi_coeffs(double x_ii, double x_ij, double x_jj, double* c, double* s) {
    if (not isclose(x_ij, 0)) {
        double tau, t, out_c;
        tau = (x_jj - x_ii)/(2*x_ij);
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
