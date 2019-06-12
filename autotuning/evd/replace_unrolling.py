#!/usr/bin/python3
import re
import sys
from gen_evd_block_vectorized import gen_evd_block_vector

def replace_unrolling(fpath, unroll_count, *args, **kwargs):
    with open(fpath, 'r') as f:
        data = f.read()
    separate_data = re.split("// Perform EVD for the block", data)
    new_function_code = gen_evd_block_vector(unroll_count)
    new_data = "// Perform EVD for the block\n".join([separate_data[0], new_function_code])

    with open(fpath, 'w') as f:
        f.write(new_data)


def main():
    if len(sys.argv) < 3:
        print('Usage: {} <fpath> <unroll_count>')
        print('<unroll_cnt> is the number of vector unrollings. For example, if it is 2, 8 elements will be loaded and processed')
        sys.exit(-1)

    fpath = sys.argv[1]
    unroll_cnt = sys.argv[2]

    replace_unrolling(fpath, unroll_cnt)


if __name__ == '__main__':
    main()
