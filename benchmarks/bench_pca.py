"""K-means clustering"""

import numpy as np
from datetime import datetime
from scikits.learn import pca as skl_pca
from pybrain.auxiliary import pca as pybrain_pca
import mdp
from mvpa.mappers.pca import PCAMapper as MVPA_PCA
from mvpa.datasets import Dataset

#
#       .. Load dataset ..
#
from load import load_data, bench
print 'Loading data ...'
X, y, T = load_data()
print 'Done, %s samples with %s features loaded into ' \
      'memory' % X.shape
n_components = 9



def bench_skl():
#
#       .. scikits.learn ..
#
    start = datetime.now()
    clf = skl_pca.RandomizedPCA(n_components=n_components)
    clf.fit(X)
    return datetime.now() - start


def bench_pybrain():
#
#       .. pybrain ..
#
    start = datetime.now()
    pybrain_pca.pca(X, n_components)
    return datetime.now() - start


def bench_mdp():
#
#       .. MDP ..
#
    start = datetime.now()
    mdp.pca(X, output_dim=n_components)
    return datetime.now() - start


def bench_mvpa():
#
#       .. PyMVPA ..
#
    start = datetime.now()
    clf = MVPA_PCA()
    print 'Warning, PyMVPA does not accept keyword to set number ' \
          'of components'
    data = Dataset(samples=X, labels=0)
    clf.train(data)
    return datetime.now() - start


if __name__ == '__main__':
    print __doc__
    print 'scikits.learn: ', bench(bench_skl), bench(bench_skl)
    print 'pybrain: ', bench(bench_pybrain), bench(bench_pybrain)
    print 'MDP: ', bench(bench_mdp), bench(bench_mdp)
    print 'PyMVPA: ', bench(bench_mvpa), bench(bench_mvpa)
