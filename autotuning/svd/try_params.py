import subprocess
from os.path import join as pathjoin
from replace_unrolling import replace_unrolling
from gen_subprocedure_vectorized import gen_subprocedure_vectorized
from gen_subprocedure_vectorized_rowwise import gen_subprocedure_vectorized_rowwise
from gen_subprocedure_vectorized_with_transpose import gen_subprocedure_vectorized_with_transpose


SVD_SRC_DIR = 'src/svd/two-sided'
SVD_PERF_SRC_DIR = 'perf/svd/two-sided'
SVD_BIN_DIR = 'bin/benchmark/svd'

subprocedure_names = ['svd_subprocedure_vectorized.cpp', 'svd_subprocedure_vectorized_rowwise.cpp', 'svd_subprocedure_vectorized_with_transpose.cpp']
subprocedure_paths = [pathjoin(SVD_SRC_DIR, name) for name in subprocedure_names]
subprocedure_gen_funcs = [gen_subprocedure_vectorized, gen_subprocedure_vectorized_rowwise, gen_subprocedure_vectorized_with_transpose]

perf_src_names = ["svd_blocked.cpp", "svd_blocked_less_copy_transposed.cpp", "svd_blocked_less_copy.cpp"]
perf_src_paths = [pathjoin(SVD_PERF_SRC_DIR, name) for name in perf_src_names]

bin_paths = [pathjoin(SVD_BIN_DIR, name[:name.find('.')].replace('_', '-')) for name in perf_src_names]

block_size_list = [8, 16, 32, 64, 128]
unrolling_cnt_list = [2, 3, 4, 5]
for subproc_path, subproc_gen_fn, perf_src_name, perf_src_path, bin_path in zip(subprocedure_paths, subprocedure_gen_funcs, perf_src_names, perf_src_paths, bin_paths):
    for block_size in block_size_list:
        for unrolling_cnt in unrolling_cnt_list:
            print(f'Func name:     {perf_src_name}')
            print(f'Block size:    {block_size}')
            print(f'Unrolling cnt: {unrolling_cnt}')

            replace_unrolling(subproc_path, subproc_gen_fn, unrolling_cnt)
            subprocess.call(['sed', '-i', f's/size_t block_size.*/size_t block_size = {block_size};/g', perf_src_path])
            subprocess.run(['./scripts/build.sh'], shell=True)
            print(f'Running benchmark {bin_path}')
            subprocess.run(['./scripts/benchmark.sh', f'{bin_path}'], shell=True)
