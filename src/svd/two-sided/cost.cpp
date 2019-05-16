#include "cost.hpp"
#include <cstddef>

size_t base_cost(size_t n, size_t n_iter) {
    size_t rot_add = 8 * n;
    size_t rot_mul = 20 * n;

    size_t svd_add = 26;
    size_t svd_mul = 26;
    size_t svd_div = 5;
    size_t svd_sqrt = 3;

    size_t loops = (n * n - n) / 2;

    size_t frobenius_add = n * n;
    size_t frobenius_mul = n * n;

    size_t add = n_iter * (loops * (rot_add + svd_add) + frobenius_add);
    size_t mult = n_iter * (loops * (rot_mul + svd_mul) + frobenius_mul);
    size_t div = n_iter * loops * svd_div;
    size_t sqrt = n_iter * loops * svd_sqrt;

    return add + mult + div + sqrt;
}

size_t base_cost_vectorized(size_t n, size_t n_iter) {
    size_t rot_add = 8 * n;
    size_t rot_mul = 16 * n;

    size_t svd_add = 26;
    size_t svd_mul = 28;
    size_t svd_div = 5;
    size_t svd_sqrt = 3;

    size_t loops = (n * n - n) / 2;

    size_t frobenius_add = n * n;
    size_t frobenius_mul = n * n;

    size_t add = n_iter * (loops * (rot_add + svd_add) + frobenius_add);
    size_t mult = n_iter * (loops * (rot_mul + svd_mul) + frobenius_mul);
    size_t div = n_iter * loops * svd_div;
    size_t sqrt = n_iter * loops * svd_sqrt;

    return add + mult + div + sqrt;
}

size_t blocked_cost_without_subprocedure(size_t n, size_t b, size_t n_iter) {
    size_t n_blocks = n / b;
    size_t loops = n_blocks * (n_blocks - 1) / 2;

    size_t single_mult_block_add = b * b * b;
    size_t single_mult_block_mult = b * b * b;
    size_t single_add = b * b;

    size_t frobenius_add = n * n;
    size_t frobenius_mult = n * n;

    size_t add = n_iter * (loops * (4 * n_blocks * (2 * single_add + 4 * single_mult_block_add) + frobenius_add));
    size_t mult = n_iter * (loops * 4 * n_blocks * 4 * single_mult_block_mult + frobenius_mult);

    return add + mult;
}

size_t blocked_less_copy_cost_without_subprocedure(size_t n, size_t b, size_t n_iter) {
    size_t n_blocks = n / b;
    size_t loops = n_blocks * (n_blocks - 1) / 2;

    size_t single_mult_block_add = b * b * b;
    size_t single_mult_block_mult = b * b * b;

    size_t frobenius_add = n * n;
    size_t frobenius_mult = n * n;

    size_t add = n_iter * (loops * 4 * n_blocks * 4 * single_mult_block_add + frobenius_add);
    size_t mult = n_iter * (loops * 4 * n_blocks * 4 * single_mult_block_mult + frobenius_mult);

    return add + mult;
}
