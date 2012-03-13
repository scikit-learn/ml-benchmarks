"""Benchmarks for coordinate-descent implementations of ElasticNet"""

#
#       .. Imports ..
#
import numpy as np
from datetime import datetime


def bench_skl(X, y, T, valid):
#
#       .. scikits.learn ..
#
    from sklearn import linear_model
    start = datetime.now()
    clf = linear_model.ElasticNet(rho=0.5, alpha=0.5)
    clf.fit(X, y)
    pred = clf.predict(T)
    delta = datetime.now() - start
    mse = np.linalg.norm(pred - valid, 2) ** 2
    return mse, delta


def bench_mlpy(X, y, T, valid):
#
#       .. MLPy ..
#
    from mlpy import ElasticNet
    start = datetime.now()
    clf = ElasticNet(tau=.5, mu=.5)
    clf.learn(X, y)
    pred = clf.pred(T)
    delta = datetime.now() - start
    mse = np.linalg.norm(pred - valid, 2) ** 2
    return mse, delta


def bench_pymvpa(X, y, T, valid):
#
#       .. PyMVPA ..
#
    from mvpa.datasets import dataset_wizard
    from mvpa.clfs import glmnet
    start = datetime.now()
    data = dataset_wizard(X, y)
    clf = glmnet.GLMNET_R(alpha=.5)
    clf.train(data)
    pred = clf.predict(T)
    delta = datetime.now() - start
    mse = np.linalg.norm(pred - valid, 2) ** 2
    return mse, delta


if __name__ == '__main__':

    import sys, misc

    # don't bother me with warnings
    import warnings; warnings.simplefilter('ignore')
    np.seterr(all='ignore')

    print __doc__ + '\n'

    if not len(sys.argv) == 2:
        print misc.USAGE
        sys.exit(-1)
    else:
        dataset = sys.argv[1]

    print 'Loading data ...'
    data = misc.load_data(dataset)
    print 'Done, %s samples with %s features loaded into ' \
          'memory' % data[0].shape

    score, res_skl = misc.bench(bench_skl, data)
    print 'scikits.learn: mean %s, std %s' % (res_skl.mean(), res_skl.std())
    print 'MSE ', score

    score, res_mlpy = misc.bench(bench_mlpy, data)
    print 'MLPy: mean %s, std %s' % (res_mlpy.mean(), res_mlpy.std())
    print 'MSE ', score

    score, res_pymvpa = misc.bench(bench_pymvpa, data)
    print 'PyMVPA: mean %s, std %s' % (res_pymvpa.mean(), res_pymvpa.std())
    print 'MSE ', score
