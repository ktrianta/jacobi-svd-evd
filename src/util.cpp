#include <math.h>
#include "util.hpp"

bool isclose(double x, double y, double eps) {
    return fabs(x - y) < eps * fabs(x + y);
}
