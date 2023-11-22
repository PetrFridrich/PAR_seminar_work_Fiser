from mpi4py import MPI
import numpy as np
from functools import reduce
from operator import add
import timeit
from numba import jit


def zero_cross(data):
  return np.sum(np.diff(np.sign(data)) != 0)


def part_iter(n, p):

    for i in range(p):
        start = (i * n) // p
        end = ((i+1) * n) // p
        yield start,end - start


def main():
   
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    number_of_replications = 100

    n_attempts = 50

    n = number_of_replications * 100_000
    N = n // size

    if rank == 0:        
        times = np.zeros(n_attempts)

        y = np.loadtxt('Data/ABPsignal.txt', delimiter='\t')[:,1]  # Loading data.
        y = (((y - np.min(y)) / (np.max(y)-np.min(y)))*2)-1  # Scaling to (-1,1).
        y = np.tile(y, number_of_replications)  # Dataset enlargement.
    else: 
        y = None

    for attempt in range(n_attempts):

        if rank == 0:
            t0 = timeit.default_timer()
        else:
            pass
        
        # Scatter
        y_part = np.empty(N)
        comm.Scatter(y, y_part, root=0)

        local_result = zero_cross(y_part)

        # Gather        
        if rank == 0:
            recvbuf = np.empty(size, dtype='i')
        else:
            recvbuf = None
        
        comm.Gather(local_result, recvbuf, root=0)

        if rank == 0:
            global_result = sum(recvbuf)
            times[attempt] = timeit.default_timer() - t0
        
    if rank == 0:
        print(times)

if __name__ == "__main__":

    main()