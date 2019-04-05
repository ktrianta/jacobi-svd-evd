import numpy as np
import sys

n = int(sys.argv[1])

data = np.random.uniform(0, 100, size=(n, n))
data = (data + data.T) / 2
e, v = np.linalg.eig(data)
idx = (-1 * e).argsort()
e = e[idx]
v = v[:, idx]

np.savetxt(sys.stdout, data, delimiter=" ", newline=" ")
np.savetxt(sys.stdout, e,    delimiter=" ", newline=" ")
np.savetxt(sys.stdout, v,    delimiter=" ", newline=" ")
