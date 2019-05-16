#pragma once
#include <cstddef>

size_t base_cost(size_t n, size_t n_iter);
size_t base_cost_vectorized(size_t n, size_t n_iter);
size_t blocked_cost_without_subprocedure(size_t n, size_t b, size_t n_iter);
size_t blocked_less_copy_cost_without_subprocedure(size_t n, size_t b, size_t n_iter);
