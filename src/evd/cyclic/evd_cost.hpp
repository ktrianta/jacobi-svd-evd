#pragma once
#include <cstddef>

size_t base_cost_evd(size_t n, size_t n_iter);
// double tol_cost(size_t n, size_t n_iter);
size_t base_cost_vectorized_evd(size_t n, size_t n_iter);
size_t outer_unroll_cost_evd(size_t n, size_t n_iter);
size_t outer_unroll_vectorize_cost_evd(size_t n, size_t n_iter);
size_t oneloop_cost_evd(size_t n, size_t n_iter);
size_t oneloop_vectorize_cost_evd(size_t n, size_t n_iter);
size_t blocked_cost_without_subprocedure_evd(size_t n, size_t b, size_t n_iter, size_t individual_block_iter);
size_t blocked_less_copy_cost_without_subprocedure_evd(size_t n, size_t b, size_t n_iter,
                                                       size_t individual_block_iter);
size_t subprocedure_cost(size_t b, size_t n_iter);
size_t blocked_cost_with_subprocedure_evd(size_t n, size_t b, size_t n_iter, size_t individual_block_iter);
size_t blocked_less_copy_cost_with_subprocedure_evd(size_t n, size_t b, size_t n_iter,
                                                    size_t individual_block_iter);
