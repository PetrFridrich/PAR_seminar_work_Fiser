from mpi4py import MPI
import numpy as np
from functools import reduce
from operator import add
import timeit


def zero_cross(data):
  prev_positive = data[0] > 0
  num_cross = 0

  for y_val in data[1:]:
    if y_val > 0 and not prev_positive:
      num_cross += 1
      prev_positive = True
    elif y_val < 0 and prev_positive:
      num_cross +=1
      prev_positive = False

  return num_cross


def part_iter(sig_len, worker_cnt):
    for i in range(worker_cnt):
        start = (i * sig_len) // worker_cnt
        end = ((i+1)*sig_len) // worker_cnt
        yield start, end


def list_partitioner(l, worker_cnt):
    return [l[max(0, start - 1): end] for start, end in part_iter(len(l), worker_cnt)]


def main():
   
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_attempts = 50    

    if rank == 0:
        
        times = np.zeros(n_attempts)

        y = np.loadtxt('Data/ABPsignal.txt', delimiter='\t')[:,1]  # Loading data.
        y = (((y - np.min(y)) / (np.max(y)-np.min(y)))*2)-1  # Scaling to (-1,1).
        y = np.tile(y, 100)  # Dataset enlargement.

    for attempt in range(n_attempts):

        if rank == 0:
            t0 = timeit.default_timer()
            y_parts = list_partitioner(y, size)
        else:
            y_parts = None

        local_y_part = comm.scatter(y_parts)
        local_result = zero_cross(local_y_part)

        local_results = comm.gather(local_result)

        if rank == 0:
            global_result = reduce(add, local_results)
            times[attempt] = timeit.default_timer() - t0
    
    if rank == 0:
       print(times)


if __name__ == "__main__":

    main()