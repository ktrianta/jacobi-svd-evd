#pragma once
#include <cstddef>

size_t base_cost(size_t n, size_t n_iter);
size_t blocked_cost(size_t n, size_t b, size_t main_iter, size_t block_iter);
