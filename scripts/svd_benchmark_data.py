import numpy as np
import sys

# Run as 'python evd_input_testdata.py <size1> <size2> > <file>'
# to write to <file> a matrix of size <size1> by <size2>.
# The first line of <file> is "<size1> <size2>\n"

assert(len(sys.argv) == 3)
m = int(sys.argv[1])
n = int(sys.argv[2])

data = np.random.uniform(-100, 100, size=(m, n))

print('{} {}'.format(m, n))
np.savetxt(sys.stdout, data, delimiter=" ", newline=" ")
