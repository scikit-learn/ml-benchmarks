"""Benchmarks for coordinate-descent implementations of ElasticNet"""

#
#       .. Imports ..
#
import numpy as np
from datetime import datetime

#
#       .. Load dataset ..
#
from load import load_data, bench
print 'Loading data ...'
X, y, T = load_data()
print 'Done, %s samples with %s features loaded into ' \
      'memory' % X.shape


def bench_skl():
#
#       .. scikits.learn ..
#
    from scikits.learn import linear_model as skl_lm
    start = datetime.now()
    skl_clf = skl_lm.ElasticNet(rho=0.5)
    skl_clf.fit(X, y)
    skl_clf.predict(T)
    return datetime.now() - start


def bench_mlpy():
#
#       .. MLPy ..
#
    from mlpy import ElasticNet as mlpy_enet
    start = datetime.now()
    mlpy_clf = mlpy_enet(tau=.5, mu=.5)
    mlpy_clf.learn(X, y)
    mlpy_clf.pred(T)
    return datetime.now() - start


def bench_pymvpa():
#
#       .. PyMVPA ..
#
    from mvpa.datasets import dataset_wizard
    from mvpa.clfs import glmnet as mvpa_glmnet
    tstart = datetime.now()
    data = dataset_wizard(X, y)
    clf = mvpa_glmnet.GLMNET_R(alpha=.5)
    clf.train(data)
    clf.predict(T)
    return datetime.now() - tstart
    

if __name__ == '__main__':

    # don't bother me with warnings
    import warnings; warnings.simplefilter('ignore')
    np.seterr(all='ignore')

    print __doc__ + '\n'

    res_skl = bench(bench_skl)
    print 'scikits.learn: mean %s, std %s' % (res_skl.mean(), res_skl.std())

    res_mlpy = bench(bench_mlpy)
    print 'MLPy: mean %s, std %s' % (res_mlpy.mean(), res_mlpy.std())

    res_pymvpa = bench(bench_pymvpa)
    print 'PyMVPA: mean %s, std %s' % (res_pymvpa.mean(), res_pymvpa.std())
