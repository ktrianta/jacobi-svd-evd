import numpy as np

n = 4
m = 5

data = np.random.randint(0,10,size=(n,m))
u, s, v = np.linalg.svd(data)

np.savetxt("svdmatrix.txt",data,delimiter= " ,",newline = ",\n")
np.savetxt("singularvalues.txt",s,delimiter= " ,",newline = ",\n")
np.savetxt("vsingularvector.txt",v, delimiter = " ,",newline = ",\n")
np.savetxt("usingularvector.txt",u, delimiter = " ,",newline = ",\n")
