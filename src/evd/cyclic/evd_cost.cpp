#include "evd_cost.hpp"
#include <cstddef>

size_t base_cost_evd(size_t n, size_t n_iter) {
    size_t loops = n * (n - 1) * 0.5;

    size_t rot_add = 4 * loops;
    size_t evd_add = 6 * n * loops;
    size_t add = n_iter * (rot_add + evd_add);

    size_t rot_mul = 3 * loops;
    size_t evd_mul = 12 * n * loops;
    size_t mul = n_iter * (rot_mul + evd_mul);

    // These two are only used in sym_jacobi_coeffs with mutually exclusive condition
    size_t div = n_iter * (3 * loops);
    size_t sqrt = n_iter * (2 * loops);

    return add + mul + div + sqrt;
}

size_t tol_cost(size_t n, size_t n_iter) {
    size_t top_ele = n * (n - 1) * 0.5;
    size_t adds = 4 + 6 * n;
    size_t muls = 3 + 12 * n;
    size_t divs = 3;
    size_t sqrt = 2;

    size_t freb_flops = 2 * (n - 1) * n;

    return n_iter * (top_ele * (adds + muls + divs + sqrt) + freb_flops);
    ;
}

size_t oneloop_cost_evd(size_t n, size_t n_iter) {
    // number of upper triangular elements
    size_t loops = n * (n - 1) * 0.5;

    size_t rot_add = 8 * loops;
    size_t evd_add = 4 * n * loops;
    size_t add = n_iter * (rot_add + evd_add);

    size_t rot_mul = 11 * loops;
    size_t evd_mul = 8 * n * loops;
    size_t mul = n_iter * (rot_mul + evd_mul);

    // These two are only used in sym_jacobi_coeffs with mutually exclusive condition
    size_t div = n_iter * (3 * loops);
    size_t sqrt = n_iter * (2 * loops);

    return add + mul + div + sqrt;
}

size_t blocked_cost_without_subprocedure_evd(size_t n, size_t b, size_t n_iter) {
    size_t n_blocks = n / b;
    size_t loops = n_blocks * (n_blocks - 1) / 2;

    size_t single_mult_block_add = b * b * b;
    size_t single_mult_block_mul = b * b * b;
    size_t single_add = b * b;

    size_t block_adds = loops * (3 * n_blocks * (2 * single_add + 4 * single_mult_block_add));
    size_t block_muls = loops * (3 * n_blocks * (4 * single_mult_block_mul));
    size_t total_ops_per_iter = (block_adds + block_muls);

    return n_iter * total_ops_per_iter;
}

size_t blocked_less_copy_cost_without_subprocedure_evd(size_t n, size_t b, size_t n_iter) {
    size_t n_blocks = n / b;
    size_t loops = n_blocks * (n_blocks - 1) / 2;

    size_t single_mult_block_add = b * b * b;
    size_t single_mult_block_mul = b * b * b;

    size_t block_adds = loops * (3 * n_blocks * (4 * single_mult_block_add));
    size_t block_muls = loops * (3 * n_blocks * (4 * single_mult_block_mul));
    size_t total_ops_per_iter = (block_adds + block_muls);

    return n_iter * total_ops_per_iter;
}

size_t subprocedure_cost(size_t b, size_t n_iter) {
    size_t loops = b * (b - 1) * 0.5;
    size_t adds = 18;
    size_t muls = 25;
    size_t divs = 5;
    size_t sqrt = 3;
    size_t vec_adds = 6 * b;
    size_t vec_muls = 12 * b;

    size_t freb_flops = 2 * (b - 1) * b;

    return n_iter * (freb_flops + loops * (sqrt + adds + muls + divs + vec_muls + vec_fmas));
}
