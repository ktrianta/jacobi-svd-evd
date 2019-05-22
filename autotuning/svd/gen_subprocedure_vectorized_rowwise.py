def gen_subprocedure_vectorized_rowwise(unroll_cnt):
    VEC_SIZE = 4
    unroll_elems = unroll_cnt * VEC_SIZE

    line_list = []
################# FIRST LOOP #################################
    line_list.append('size_t k = 0;')
    for u in range(unroll_cnt):
        line_list.append(f'__m256d vi_{u};')
        line_list.append(f'__m256d vj_{u};')
        line_list.append(f'__m256d left_{u};')
        line_list.append(f'__m256d right_{u};')
    line_list.append('__m256d v_c = _mm256_set1_pd(cf.c1);')
    line_list.append('__m256d v_s = _mm256_set1_pd(cf.s1);')
    line_list.append('__m256d v_s1k = _mm256_set1_pd(s1k);')
    line_list.append('__m256d v_c1k = _mm256_set1_pd(c1k);')
    line_list.append(f'for (; k + {unroll_elems-1} < n; k += {unroll_elems}) {{')
    for u in range(unroll_cnt):
        line_list.append(f'    vi_{u} = _mm256_set_pd(B[n * (k + {u*VEC_SIZE}) + i]')
        for m in range(1, VEC_SIZE):
            line_list[-1] += f', B[n * (k + {u*VEC_SIZE + m}) + i]'
        line_list[-1] += ');'
    for u in range(unroll_cnt):
        line_list.append(f'    vj_{u} = _mm256_set_pd(B[n * (k + {u*VEC_SIZE}) + j]')
        for m in range(1, VEC_SIZE):
            line_list[-1] += f', B[n * (k + {u*VEC_SIZE + m}) + j]'
        line_list[-1] += ');'
    line_list.append('')
    for u in range(unroll_cnt):
        line_list.append(f'    left_{u} = _mm256_mul_pd(v_s, vj_{u});')
    for u in range(unroll_cnt):
        line_list.append(f'    left_{u} = _mm256_fmsub_pd(v_c, vi_{u}, left_{u});')
    line_list.append('')
    for u in range(unroll_cnt):
        line_list.append(f'    right_{u} = _mm256_mul_pd(v_c1k, vj_{u});')
    for u in range(unroll_cnt):
        line_list.append(f'    right_{u} = _mm256_fmadd_pd(v_s1k, vi_{u}, right_{u});')
    line_list.append('')
    for u in range(unroll_cnt):
        line_list.append(f'    double* left_{u}_ptr = (double*) &left_{u};')
    for u in range(unroll_cnt):
        line_list.append(f'    double* right_{u}_ptr = (double*) &right_{u};')
    for u in range(unroll_cnt):
        for m in range(VEC_SIZE):
            line_list.append(f'    B[n * (k + {u*VEC_SIZE+m}) + i] = left_{u}_ptr[{VEC_SIZE-1-m}];')
    for u in range(unroll_cnt):
        for m in range(VEC_SIZE):
            line_list.append(f'    B[n * (k + {u*VEC_SIZE+m}) + j] = right_{u}_ptr[{VEC_SIZE-1-m}];')
    line_list.append('}')
    line_list.append('for (; k < n; ++k) {')
    line_list.append('    double b_ik = B[n * k + i];')
    line_list.append('    double b_jk = B[n * k + j];')
    line_list.append('    double left = cf.c1 * b_ik - cf.s1 * b_jk;')
    line_list.append('    double right = s1k * b_ik + c1k * b_jk;')
    line_list.append('    B[n * k + i] = left;')
    line_list.append('    B[n * k + j] = right;')
    line_list.append('}')
    line_list.append('')

################# SECOND LOOP #################################
    line_list.append('k = 0;')
    line_list.append('v_c = _mm256_set1_pd(cf.c2);')
    line_list.append('v_s = _mm256_set1_pd(cf.s2);')
    line_list.append(f'for (; k + {unroll_elems-1} < n; k += {unroll_elems}) {{')
    for u in range(unroll_cnt):
        line_list.append(f'    vi_{u} = _mm256_load_pd(B + n * i + (k + {u*VEC_SIZE}));')
    for u in range(unroll_cnt):
        line_list.append(f'    vj_{u} = _mm256_load_pd(B + n * j + (k + {u*VEC_SIZE}));')
    line_list.append('')
    for u in range(unroll_cnt):
        line_list.append(f'    left_{u} = _mm256_mul_pd(v_s, vj_{u});')
    for u in range(unroll_cnt):
        line_list.append(f'    left_{u} = _mm256_fmsub_pd(v_c, vi_{u}, left_{u});')
    line_list.append('')
    for u in range(unroll_cnt):
        line_list.append(f'    right_{u} = _mm256_mul_pd(v_c, vj_{u});')
    for u in range(unroll_cnt):
        line_list.append(f'    right_{u} = _mm256_fmadd_pd(v_s, vi_{u}, right_{u});')
    line_list.append('')
    for u in range(unroll_cnt):
        line_list.append(f'    _mm256_store_pd(B + n * i + (k + {u*VEC_SIZE}), left_{u});')
    for u in range(unroll_cnt):
        line_list.append(f'    _mm256_store_pd(B + n * j + (k + {u*VEC_SIZE}), right_{u});')
    line_list.append('}')
    line_list.append('for (; k < n; ++k) {')
    line_list.append('    double b_ki = B[n * i + k];')
    line_list.append('    double b_kj = B[n * j + k];')
    line_list.append('    double left = cf.c2 * b_ki - cf.s2 * b_kj;')
    line_list.append('    double right = cf.s2 * b_ki + cf.c2 * b_kj;')
    line_list.append('    B[n * i + k] = left;')
    line_list.append('    B[n * j + k] = right;')
    line_list.append('}')
    line_list.append('')


################# THIRD LOOP #################################
    line_list.append('k = 0;')
    line_list.append('v_c = _mm256_set1_pd(cf.c1);')
    line_list.append('v_s = _mm256_set1_pd(cf.s1);')
    line_list.append(f'for (; k + {unroll_elems-1} < n; k += {unroll_elems}) {{')
    for u in range(unroll_cnt):
        line_list.append(f'    vi_{u} = _mm256_load_pd(U + n * i + (k + {u*VEC_SIZE}));')
    for u in range(unroll_cnt):
        line_list.append(f'    vj_{u} = _mm256_load_pd(U + n * j + (k + {u*VEC_SIZE}));')
    line_list.append('')
    for u in range(unroll_cnt):
        line_list.append(f'    left_{u} = _mm256_mul_pd(v_s, vj_{u});')
    for u in range(unroll_cnt):
        line_list.append(f'    left_{u} = _mm256_fmsub_pd(v_c, vi_{u}, left_{u});')
    line_list.append('')
    for u in range(unroll_cnt):
        line_list.append(f'    right_{u} = _mm256_mul_pd(v_c1k, vj_{u});')
    for u in range(unroll_cnt):
        line_list.append(f'    right_{u} = _mm256_fmadd_pd(v_s1k, vi_{u}, right_{u});')
    line_list.append('')
    for u in range(unroll_cnt):
        line_list.append(f'    _mm256_store_pd(U + n * i + (k + {u*VEC_SIZE}), left_{u});')
    for u in range(unroll_cnt):
        line_list.append(f'    _mm256_store_pd(U + n * j + (k + {u*VEC_SIZE}), right_{u});')
    line_list.append('}')
    line_list.append('for (; k < n; ++k) {')
    line_list.append('    double u_ki = U[n * i + k];')
    line_list.append('    double u_kj = U[n * j + k];')
    line_list.append('    double left = cf.c1 * u_ki - cf.s1 * u_kj;')
    line_list.append('    double right = s1k * u_ki + c1k * u_kj;')
    line_list.append('    U[n * i + k] = left;')
    line_list.append('    U[n * j + k] = right;')
    line_list.append('}')
    line_list.append('')

################# FOURTH LOOP #################################
    line_list.append('k = 0;')
    line_list.append('v_c = _mm256_set1_pd(cf.c2);')
    line_list.append('v_s = _mm256_set1_pd(cf.s2);')
    line_list.append(f'for (; k + {unroll_elems-1} < n; k += {unroll_elems}) {{')
    for u in range(unroll_cnt):
        line_list.append(f'    vi_{u} = _mm256_load_pd(V + n * i + (k + {u*VEC_SIZE}));')
    for u in range(unroll_cnt):
        line_list.append(f'    vj_{u} = _mm256_load_pd(V + n * j + (k + {u*VEC_SIZE}));')
    line_list.append('')
    for u in range(unroll_cnt):
        line_list.append(f'    left_{u} = _mm256_mul_pd(v_s, vj_{u});')
    for u in range(unroll_cnt):
        line_list.append(f'    left_{u} = _mm256_fmsub_pd(v_c, vi_{u}, left_{u});')
    line_list.append('')
    for u in range(unroll_cnt):
        line_list.append(f'    right_{u} = _mm256_mul_pd(v_c, vj_{u});')
    for u in range(unroll_cnt):
        line_list.append(f'    right_{u} = _mm256_fmadd_pd(v_s, vi_{u}, right_{u});')
    line_list.append('')
    for u in range(unroll_cnt):
        line_list.append(f'    _mm256_store_pd(V + n * i + (k + {u*VEC_SIZE}), left_{u});')
    for u in range(unroll_cnt):
        line_list.append(f'    _mm256_store_pd(V + n * j + (k + {u*VEC_SIZE}), right_{u});')
    line_list.append('}')
    line_list.append('for (; k < n; ++k) {')
    line_list.append('    double v_ki = V[n * i + k];')
    line_list.append('    double v_kj = V[n * j + k];')
    line_list.append('    double left = cf.c2 * v_ki - cf.s2 * v_kj;')
    line_list.append('    double right = cf.s2 * v_ki + cf.c2 * v_kj;')
    line_list.append('    V[n * i + k] = left;')
    line_list.append('    V[n * j + k] = right;')
    line_list.append('}')

    return line_list
