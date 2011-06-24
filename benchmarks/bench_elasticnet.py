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
    from scikits.learn import linear_model as skl_lm
    start = datetime.now()
    skl_clf = skl_lm.ElasticNet(rho=0.5)
    skl_clf.fit(X, y)
    mse = np.linalg.norm(
        skl_clf.predict(T) - valid, 2) ** 2
    return mse, datetime.now() - start


def bench_mlpy(X, y, T, valid):
#
#       .. MLPy ..
#
    from mlpy import ElasticNet as mlpy_enet
    start = datetime.now()
    mlpy_clf = mlpy_enet(tau=.5, mu=.5)
    mlpy_clf.learn(X, y)
    mse = np.linalg.norm(
        mlpy_clf.pred(T) - valid, 2) ** 2
    return mse, datetime.now() - start


def bench_pymvpa(X, y, T, valid):
#
#       .. PyMVPA ..
#
    from mvpa.datasets import dataset_wizard
    from mvpa.clfs import glmnet as mvpa_glmnet
    tstart = datetime.now()
    data = dataset_wizard(X, y)
    clf = mvpa_glmnet.GLMNET_R(alpha=.5)
    clf.train(data)
    mse = np.linalg.norm(
        clf.predict(T) - valid, 2) ** 2
    return mse, datetime.now() - tstart


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

##    score, res_skl = misc.bench(bench_skl, data)
##    print 'scikits.learn: mean %s, std %s' % (res_skl.mean(), res_skl.std())

    score, res_mlpy = misc.bench(bench_mlpy, data)
    print 'MLPy: mean %s, std %s' % (res_mlpy.mean(), res_mlpy.std())

##    score, res_pymvpa = misc.bench(bench_pymvpa, data)
##    print 'PyMVPA: mean %s, std %s' % (res_pymvpa.mean(), res_pymvpa.std())
