
import numpy as np
import os

def load_data():

    f = open(os.path.dirname(__file__) + '/data/madelon_train.data')
    X = np.fromfile(f, dtype=np.float64, sep=' ')
    f.close()
    X = X.reshape(-1, 500)

    f = open(os.path.dirname(__file__) + '/data/madelon_train.labels')
    y = np.fromfile(f, dtype=np.int32, sep=' ')
    f.close()

    f = open(os.path.dirname(__file__) + '/data/madelon_test.data')
    T = np.fromfile(f, dtype=np.float64, sep=' ')
    f.close()
    T = T.reshape(-1, 500)

    return  X, y, T
    

def dtime_to_seconds(dtime):
    return dtime.seconds + (dtime.microseconds * 1e-6)


def bench(func, n=10):
    """
    Benchmark a given function. The function is executed n times and
    its output is expected to be of type datetime.datetime.

    All values are converted to seconds and returned in an array.

    Returns
    -------
    D : array, size=n-2
    """
    assert n > 2
    try:
        time = [dtime_to_seconds(func()) for i in range(n)]
        # remove extremal values
        time.pop(np.argmax(time))
        time.pop(np.argmin(time))
    except Exception as detail:
        print '%s error in function %s: ' % (repr(detail), func)
        time = []
    return np.array(time)
