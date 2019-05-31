import sys
from gen_subprocedure_vectorized import gen_subprocedure_vectorized
from gen_subprocedure_vectorized_rowwise import gen_subprocedure_vectorized_rowwise
from gen_subprocedure_vectorized_with_transpose import gen_subprocedure_vectorized_with_transpose


def n_indent_spaces(line):
    cnt = 0
    for c in line:
        if not c.isspace():
            break
        cnt += 1
    return cnt


def replace_unrolling(fpath, generator_fn, *args, **kwargs):
    with open(fpath, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    beg = 0
    for i, line in enumerate(lines):
        if line.endswith('size_t k = 0;'):
            beg = i
            break

    n_spaces = n_indent_spaces(lines[beg])

    for i, line in enumerate(lines[beg:]):
        if line and n_indent_spaces(line) < n_spaces:
            end = beg + i
            break

    generated_lines = generator_fn(*args, **kwargs)
    lines[beg:end] = generated_lines
    end = beg + len(generated_lines)

    indentation = ' ' * n_spaces
    for i in range(beg, end):
        if lines[i]:
            lines[i] = '{}{}'.format(indentation, lines[i])

    with open(fpath, 'w') as f:
        for line in lines:
            f.write('{}\n'.format(line))


def main():
    if len(sys.argv) < 4:
        print('Usage: {} <fpath> <subprocedure_name> <unroll_count>')
        print('<subprocedure_name> is one of vectorized, vectorized_rowwise or vectorized_with_transpose')
        print('<unroll_cnt> is the number of vector unrollings. For example, if it is 2, 8 elements will be loaded and processed')
        sys.exit(-1)

    fpath = sys.argv[1]
    subprocedure_name = sys.argv[2]
    unroll_cnt = int(sys.argv[3])

    if subprocedure_name == 'vectorized':
        gen_fn = gen_subprocedure_vectorized
    elif subprocedure_name == 'vectorized_rowwise':
        gen_fn = gen_subprocedure_vectorized_rowwise
    elif subprocedure_name == 'vectorized_with_transpose':
        gen_fn = gen_subprocedure_vectorized_with_transpose
    else:
        print('Incorrect subprocedure name. Run with no arguments to see help.')
        sys.exit(-2)

    replace_unrolling(fpath, gen_fn, unroll_cnt)


if __name__ == '__main__':
    main()
