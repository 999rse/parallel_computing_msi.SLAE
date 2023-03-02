from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 1000  # matrix size
epsilon = 0.0001
max_iterations = int(1e6)

# read A and b matrices from files
A = np.zeros((N, N))
b = np.zeros(N)
if rank == 0:
    with open("A.txt", "r") as f:
        for i in range(N):
            A[i, :] = [float(x) for x in f.readline().split()]
    with open("b.txt", "r") as f:
        b = [float(x) for x in f.readline().split()]

start_time = time.time()

# broadcast A and b matrices to all processes
A = comm.bcast(A, root=0)
b = comm.bcast(b, root=0)

# calculate diagonal dominant matrix
subsum = np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))
A = np.diag(2.0 * subsum) + A

# initialize solution vector x
x = np.zeros(N)

# parallelize iterations
for k in range(max_iterations):
    # calculate new approximation of x
    x_new = np.zeros(N)
    for i in range(rank, N, size):
        sum = np.dot(A[i, :], x) - A[i, i] * x[i]
        x_new[i] = (b[i] - sum) / A[i, i]

    # reduce norm across all processes
    norm = np.linalg.norm(x_new - x)
    norm = comm.allreduce(norm, op=MPI.SUM)

    # check for convergence
    if np.sqrt(norm) < epsilon:
        break

    # update x
    x = x_new.copy()

# print solution on rank 0 process
prevN = N - 2
if rank == 0:
    print("MPI-Python")
    for i in range(prevN,N):
        print("x[{}] = {}".format(i, x[i]))

# finalize MPI
MPI.Finalize()

print("Elapsed time = %s seconds" % (time.time() - start_time))
