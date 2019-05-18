#include "svd.hpp"
#include <assert.h>
#include "cost.hpp"
#include "matrix.hpp"
#include "svd_subprocedure.hpp"
#include "types.hpp"

size_t svd(struct matrix_t Amat, struct matrix_t Bmat, struct matrix_t Umat, struct matrix_t Vmat) {
    assert(Amat.rows == Amat.cols);  // Matrix A should be square
    assert(Amat.rows == Bmat.rows && Amat.cols == Bmat.cols);
    assert(Amat.rows == Umat.rows && Amat.cols == Umat.cols);
    assert(Amat.rows == Vmat.rows && Amat.cols == Vmat.cols);

    matrix_copy(Bmat, Amat);
    size_t iter = svd_subprocedure(Bmat, Umat, Vmat);
    return base_cost(Amat.rows, iter);
}
