#!/usr/bin/python3
# Usage: from the root directory executed as "python3 autotuning/evd/try_params.py"
import subprocess
import sys
from os.path import join as pathjoin
from replace_unrolling import replace_unrolling
from gen_evd_block_vectorized import gen_evd_block_vector

EVD_SRC_DIR = 'src/evd/cyclic'
EVD_PERF_SRC_DIR = 'perf/evd/cyclic'
EVD_BIN_DIR = 'bin/benchmark/evd'

file_names = ['evd_cyclic_blocked_vectorize.cpp', 'evd_cyclic_blocked_vectorize.cpp']
file_paths = [pathjoin(EVD_SRC_DIR, name) for name in file_names]

perf_src_names = ["blocked_vectorized_evd.cpp", "blocked_vectorized_evd_less_copy.cpp"]
perf_src_paths = [pathjoin(EVD_PERF_SRC_DIR, name) for name in perf_src_names]

bin_paths = [pathjoin(EVD_BIN_DIR, name) for name in ["evd-blocked-vectorized", "evd-blocked-less-copy-vectorized"]]

block_size_list = [16, 32, 64, 128]
unrolling_cnt_list = [1, 2, 3]
for file_path, perf_src_name, perf_src_path, bin_path in zip(file_paths, perf_src_names, perf_src_paths, bin_paths):
    for block_size in block_size_list:
        for unrolling_cnt in unrolling_cnt_list:
            print(f'Func name:     {perf_src_name}')
            print(f'Block size:    {block_size}')
            print(f'Unrolling cnt: {unrolling_cnt}')
            sys.stdout.flush()

            replace_unrolling(file_path, unrolling_cnt)
            subprocess.call(['sed', '-i', f's/size_t block_size.*/size_t block_size = {block_size};/g', perf_src_path])
            # Increase the epoch count when unroll count > 1 if more precison is required by uncommenting the lines below
            #if unrolling_cnt in [2, 3]:
                #subprocess.call(['sed', '-i', f's/size_t n_iter.*/size_t n_iter = 20;/g', perf_src_path])
                #subprocess.call(['sed', '-i', f's/size_t individual_block_iter.*/size_t individual_block_iter = 10;/g', perf_src_path])
            subprocess.run(['./scripts/build.sh'], shell=True)
            print(f'Running benchmark {bin_path}')
            subprocess.call(['./scripts/benchmark.sh', f'{bin_path}'])
