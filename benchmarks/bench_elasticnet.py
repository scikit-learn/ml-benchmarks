"""Benchmarks for coordinate-descent implementations of ElasticNet"""

#
#       .. Imports ..
#
from datetime import datetime
from scikits.learn import linear_model as skl_lm
from mlpy import ElasticNet as mlpy_enet
from mvpa.datasets import Dataset
from mvpa.clfs import glmnet as mvpa_glmnet

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
    start = datetime.now()
    skl_clf = skl_lm.ElasticNet(rho=0.5)
    skl_clf.fit(X, y)
    skl_clf.predict(T)
    return datetime.now() - start


def bench_mlpy():
#
#       .. MLPy ..
#
    start = datetime.now()
    mlpy_clf = mlpy_enet(tau=.5, mu=.5)
    mlpy_clf.learn(X, y)
    mlpy_clf.pred(T)
    return datetime.now() - start


def bench_pymvpa():
#
#       .. PyMVPA ..
#
    tstart = datetime.now()
    data = Dataset(samples=X, labels=y)
    clf = mvpa_glmnet.GLMNET_R(alpha=.5)
    clf.train(data)
    clf.predict(T)
    return datetime.now() - tstart
    

if __name__ == '__main__':
    print __doc__
    print 'scikits.learn: ', bench(bench_skl)
    print 'MLPy: ', bench(bench_mlpy)
    print 'PyMVPA: ', bench(bench_pymvpa)
