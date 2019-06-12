import numpy as np
import sys

# Run as 'python evd_input_testdata.py <size> > <file>'
# to write to <file> a symetric matrix of size <size>.
# The first line of <file> is "<size>\n"

assert(len(sys.argv) == 2)
n = int(sys.argv[1])

data = np.random.uniform(0, 100, size=(n, n))
data = (data + data.T) / 2

print(n)
np.savetxt(sys.stdout, data, delimiter=" ", newline=" ")
