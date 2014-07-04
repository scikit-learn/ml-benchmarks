
import numpy as np
import os

def load_data(dataset):
    """"
    Parameters
    ----------

    dataset : string
        Which dataset to load. Currently can be "madeon" or "arcene"
    """

    f = open(os.path.dirname(__file__) + '/data/%s_train.data' % dataset)
    X = np.fromfile(f, dtype=np.float64, sep=' ')
    f.close()

    f = open(os.path.dirname(__file__) + '/data/%s_train.labels' % dataset)
    y = np.fromfile(f, dtype=np.int32, sep=' ')
    f.close()

    f = open(os.path.dirname(__file__) + '/data/%s_test.data' % dataset)
    T = np.fromfile(f, dtype=np.float64, sep=' ')
    f.close()

    f = open(os.path.dirname(__file__) + '/data/%s_test.labels' % dataset)
    valid = np.fromfile(f, dtype=np.float64, sep=' ')
    f.close()

    if dataset == 'madelon':
        X = X.reshape(-1, 500)
        T = T.reshape(-1, 500)
    elif dataset == 'arcene':
        X = X.reshape(-1, 10000)
        T = T.reshape(-1, 10000)

    return  X, y, T, valid


def dtime_to_seconds(dtime):
    return dtime.seconds + (dtime.microseconds * 1e-6)

def bench(func, data, n=10):
    """
    Benchmark a given function. The function is executed n times and
    its output is expected to be of type datetime.datetime.

    All values are converted to seconds and returned in an array.

    Parameters
    ----------
    func: function to benchmark

    data: tuple (X, y, T, valid) containing training (X, y) and validation (T, valid) data.

    Returns
    -------
    D : array, size=n-2
    """
    assert n > 2
    score = np.inf
    try:
        time = []
        for i in range(n):
            score, t = func(*data)
            time.append(dtime_to_seconds(t))
        # remove extremal values
        time.pop(np.argmax(time))
        time.pop(np.argmin(time))
    except Exception as detail:
        print '%s error in function %s: ' % (repr(detail), func)
        time = []
    return score, np.array(time)

USAGE = """usage: python %s dataset

where dataset is one of {madelon, arcene}
"""