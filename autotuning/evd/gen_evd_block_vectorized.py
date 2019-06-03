#!/usr/bin/python3
import re
import sys

def vec_declarations(n, mat_list):
    dec_list = []
    for i in range(n):
        j = str(i)
        for m in mat_list:
            dec_list.append("__m256d " + m + "_row" + j + ", " + m + "_col" + j + ", " + \
                            m + "_rcopy" + j + ";")
        dec_list.append("__m256d sin_row" + j + ", sin_col" + j + ";")
    return dec_list

def vec_init_col(n, mat, size, inv):
    init_list = []
    index_list = []
    for i in range(n):
        j = str(i)
        for k in range(4):
             index_list.append(mat + "[" + size + " * " + inv + " + " + size + " * " + \
                               str(i * 4 + k) + " + " + "row]")
        init_list.append(mat + "_row" + j + " = _mm256_set_pd(" + ", ".join(index_list) + ");")
        index_list.clear()
        init_list.append(mat + "_rcopy" + j + " = " + mat + "_row" + j + ";")
        for k in range(4):
             index_list.append(mat + "[" + size + " * " + inv + " + " + size + " * " + \
                               str(i * 4 + k) + " + " + "col]")
        init_list.append(mat + "_col" + j + " = _mm256_set_pd(" + ", ".join(index_list) + ");")
        index_list.clear()
    return init_list

def vec_store_col(n, mat, var, size, inv):
    init_list = []
    index_list = []
    for i in range(n):
        j = str(i)
        index_list.append("double* " + mat + "_" + var + "_updated" + j + \
                          " = (double *) &" + mat + "_" + var + j + ";")
        for k in range(4):
            index_list.append(mat + "[" + size + " * " + inv + " + " + size + " * " + str(i * 4 + k) + " + " + \
                              var + "] = " + mat + "_" + var + "_updated" + j + "[" + str(i * 4 + 3 - k) + "];")
        init_list.extend(index_list)
        index_list.clear()
    return init_list

def vec_init_row(n, mat, size, inv):
    init_list = []
    init_list.extend(load_intrinsic(n, mat, "row", size, inv))
    for i in range(n):
        j = str(i)
        init_list.append(mat + "_rcopy" + j + " = " + mat + "_row" + j + ";")
    init_list.extend(load_intrinsic(n, mat, "col", size, inv))
    return init_list

def vec_store_row(n, mat, var, size, inv):
    init_list = []
    for i in range(n):
        j = str(i)
        init_list.append("_mm256_store_pd(" + mat + " + " + size + " * " + var + " + " + inv + \
                         " + " + str(i * 4) + ", " + mat + "_" + var + j + ");")
    return init_list

def two_opr_intrinsic(n, opr, opt, ip1, ip2):
    two_opr_list = []
    for i in range(n):
        j = str(i)
        two_opr_list.append(opt + j + " = " + "_mm256_" + opr + "_pd(" + ip1 + j + ", " + ip2 + ");")
    return two_opr_list

def three_opr_intrinsic(n, opr, opt, ip1, ip2, ip3):
    three_opr_list = []
    for i in range(n):
        j = str(i)
        three_opr_list.append(opt + j + " = " + "_mm256_" + opr + "_pd(" + ip1 + j + \
                              ", " + ip2 + ", " + ip3 + j + ");")
    return three_opr_list

def load_intrinsic(n, mat, var, size, inv):
    load_list = []
    for i in range(n):
        j = str(i)
        load_list.append(mat + "_" + var + j + " = " + "_mm256_load_pd(" + mat + " + " + \
                         size + " * " + var + " + " + inv + " + " + str(i * 4) + ");")
    return load_list

def gen_evd_col_operations(n, mat, size, inv):
    instr_list = []
    instr_list.extend(vec_declarations(n, [mat]))
    instr_list.extend(vec_init_col(n, mat, size, inv))
    instr_list.extend(two_opr_intrinsic(n, "mul", "sin_col", mat+"_col", "sin_vec"))
    instr_list.extend(three_opr_intrinsic(n, "fmsub", mat+"_row", mat+"_row", "cos_vec", "sin_col"))
    instr_list.extend(vec_store_col(n, mat, "row", size, inv))
    instr_list.extend(two_opr_intrinsic(n, "mul", "sin_row", mat+"_rcopy", "sin_vec"))
    instr_list.extend(three_opr_intrinsic(n, "fmadd", mat+"_col", mat+"_col", "cos_vec", "sin_row"))
    instr_list.extend(vec_store_col(n, mat, "col", size, inv))
    return instr_list

def gen_evd_row_operations(n, mat_list, size, inv):
    instr_list = []
    instr_list.extend(vec_declarations(n, mat_list))
    for mat in mat_list:
        instr_list.extend(vec_init_row(n, mat, size, inv))
        instr_list.extend(two_opr_intrinsic(n, "mul", "sin_col", mat+"_col", "sin_vec"))
        instr_list.extend(three_opr_intrinsic(n, "fmsub", mat+"_row", mat+"_row", "cos_vec", "sin_col"))
        instr_list.extend(vec_store_row(n, mat, "row", size, inv))
        instr_list.extend(two_opr_intrinsic(n, "mul", "sin_row", mat+"_rcopy", "sin_vec"))
        instr_list.extend(three_opr_intrinsic(n, "fmadd", mat+"_col", mat+"_col", "cos_vec", "sin_row"))
        instr_list.extend(vec_store_row(n, mat, "col", size, inv))
    return instr_list

def gen_evd_block_vector(unroll_count):
    function_code = \
'''static size_t evd_block_vector(struct matrix_t Amat, struct matrix_t Vmat, size_t epoch) {
    assert(Amat.rows == Amat.cols);
    double* A = Amat.ptr;

    double* V = Vmat.ptr;
    const size_t m = Amat.rows;

    matrix_identity(Vmat);

    for (size_t ep = 1; ep <= epoch; ep++) {
        double cos_t, sin_t;

        for (size_t row = 0; row < m; row++) {
            for (size_t col = row + 1; col < m; col++) {
                size_t n = m;
                __m256d sin_vec, cos_vec;

                // Compute cos_t and sin_t for the rotation
                sym_jacobi_coeffs(A[row * m + row], A[row * m + col], A[col * m + col], &cos_t, &sin_t);

                sin_vec = _mm256_set1_pd(sin_t);
                cos_vec = _mm256_set1_pd(cos_t);

                if (m % URC != 0) n = m - (m % URC);

                // Compute the eigen values by updating the columns until convergence
                for (size_t i = 0; i < n; i += URC) {
                    // UNROLLED_COL_OPS
                }

                if (m % URC != 0) {
                    for (size_t i = 0; i < m - n; i++) {
                        double A_i_r = A[m * (n + i) + row];
                        A[m * (n + i) + row] = cos_t * A[m * (n + i) + row] - sin_t * A[m * (n + i) + col];
                        A[m * (n + i) + col] = cos_t * A[m * (n + i) + col] + sin_t * A_i_r;
                    }
                }

                // Compute the eigen values by updating the rows until convergence
                for (size_t i = 0; i < n; i += URC) {
                    // UNROLLED_ROW_OPS
                }

                if (m % URC != 0) {
                    for (size_t i = 0; i < m - n; i++) {
                        double A_r_i = A[m * row + n + i];
                        A[m * row + n + i] = cos_t * A[m * row + n + i] - sin_t * A[m * col + n + i];
                        A[m * col + n + i] = cos_t * A[m * col + n + i] + sin_t * A_r_i;

                        double V_r_i = V[m * row + n + i];
                        V[m * row + n + i] = cos_t * V[m * row + n + i] - sin_t * V[m * col + n + i];
                        V[m * col + n + i] = cos_t * V[m * col + n + i] + sin_t * V_r_i;
                    }
                }
            }
        }
    }

    matrix_transpose({V, m, m}, {V, m, m});
    return base_cost_evd(m, epoch);
}'''

    # replace loop increments based on unrolling
    incr = str(4 * int(unroll_count))
    function_code = re.sub("URC", incr, function_code)

    # Add column operations
    col_ops = gen_evd_col_operations(int(unroll_count), "A", "m", "i")
    col_ops = [col_ops[0]] + [20 * ' ' + inst for inst in col_ops[1:]]
    function_code = re.sub("// UNROLLED_COL_OPS", "\n".join(col_ops), function_code)

    # Add row row_operations
    row_ops = gen_evd_row_operations(int(unroll_count), ["A", "V"], "m", "i")
    row_ops = [row_ops[0]] + [20 * ' ' + inst for inst in row_ops[1:]]
    function_code = re.sub("// UNROLLED_ROW_OPS", "\n".join(row_ops), function_code)

    return function_code


if __name__ == '__main__':
    unroll_val = input("Enter the unroll count (typically < 5)\n")
    if not str.isnumeric(unroll_val):
        print("Invalid value")
        sys.exit(-1)
    print("Unrolled evd_block_vector() for the entered unroll_count:\n")
    c = gen_evd_block_vector(int(unroll_val))
    print(c)
