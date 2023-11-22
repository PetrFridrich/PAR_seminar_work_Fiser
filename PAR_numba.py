from numba import jit, prange, set_num_threads
import timeit
import numpy as np


@jit(nopython=True, parallel=True)
def zero_cross(data):
    prev_positive = data[0] > 0
    num_cross = 0

    for i in prange(1,data.shape[0]):
        if data[i] > 0 and not prev_positive:
            num_cross = num_cross + 1
            prev_positive = True
        elif data[i] < 0 and prev_positive:
            num_cross = num_cross + 1
            prev_positive = False

    return num_cross


def main():

    n_attempts = 50

    y = np.loadtxt('Data/ABPsignal.txt', delimiter='\t')[:,1]  # Loading data.
    y = (((y - np.min(y)) / (np.max(y)-np.min(y)))*2)-1  # Scaling to (-1,1).
    y = np.tile(y, 100)  # Dataset enlargement.

    times = np.zeros(n_attempts)

    for attempt in range(n_attempts):
        t0 = timeit.default_timer()
        _ = zero_cross(y)
        t1 = timeit.default_timer()

        times[attempt] = t1-t0

    print(times)


if __name__ == "__main__":

    set_num_threads(8)

    main()