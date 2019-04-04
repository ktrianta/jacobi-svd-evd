import numpy as np
import sys

assert(len(sys.argv) == 3)
m = int(sys.argv[1])
n = int(sys.argv[2])

data = np.random.uniform(-100, 100, size=(m, n))
u, s, v = np.linalg.svd(data, full_matrices=False)

np.savetxt(sys.stdout, data, delimiter=" ", newline=" ")
np.savetxt(sys.stdout, s,    delimiter=" ", newline=" ")
np.savetxt(sys.stdout, u,    delimiter=" ", newline=" ")
np.savetxt(sys.stdout, v,    delimiter=" ", newline=" ")
