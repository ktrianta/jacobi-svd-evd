#pragma once
#include <float.h>

/**
 * Compare two double precision floating point values and return true if they
 * are close in the relative scale.
 *
 * Formally, the function does true if |x - y| < eps * |x + y| is satisfied.
 *
 * @param x Double value.
 * @param y Double value.
 * @param eps Epsilon value to check for closeness.
 */
bool isclose(double x, double y, double eps = DBL_EPSILON);
