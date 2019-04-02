import numpy as np
import sys

n = int(sys.argv[1])

data = np.random.randint(0,10,size=(n,n))
data = (data + data.T)/2
e, v = np.linalg.eig(data)
idx = (-1*e).argsort()
e = e[idx]
v = v[:,idx]

np.savetxt("evdmatrix.txt",data,delimiter= " ,", newline = ",\n")
np.savetxt("eigenvalues.txt",e,delimiter= " ,",newline = ",\n")
np.savetxt("eigenvectors.txt",v, delimiter = " ,",newline = ",\n")
