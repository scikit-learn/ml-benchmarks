"""K-means clustering"""

import numpy as np
from datetime import datetime
#from shogun.Distance import EuclidianDistance
#from shogun.Features import RealFeatures
#from shogun.Clustering import KMeans
from scikits.learn import cluster as skl_cluster
from scikits.learn.cluster.k_means_ import _e_step
#from pybrain.auxiliary import kmeans as pybrain_kmeans
#from mlpy import Kmeans as mlpy_Kmeans
from mdp.nodes import KMeansClassifier as mdp_KMeans


#
#       .. Load dataset ..
#
from load import load_data, bench
print 'Loading data ...'
X, y, T = load_data()
print 'Done, %s samples with %s features loaded into ' \
      'memory' % X.shape
n_components = 9
k = 9


def bench_shogun():
#
#       .. Shogun ..
#
    start = datetime.now()
    feat = RealFeatures(X.T)
    distance=EuclidianDistance(feat, feat)
    clf=KMeans(k, distance)
    clf.train()
    return datetime.now() - start


def bench_skl():
#
#       .. scikits.learn ..
#
    start = datetime.now()
    clf = skl_cluster.KMeans(k=k, n_init=1)
    clf.fit(X)
    return datetime.now() - start


def bench_pybrain():
#
#       .. pybrain ..
#
    start = datetime.now()
    pybrain_kmeans.kmeanspp(X, k)
    return datetime.now() - start


def bench_mlpy():
#
#       .. MLPy ..
#
    start = datetime.now()
    clf = mlpy_Kmeans(k)
    clf.compute(X)
    return datetime.now() - start


def bench_mdp():
#
#       .. MDP ..
#
    start = datetime.now()
    clf = mdp_KMeans(k)
    clf.label(X)
    return datetime.now() - start



if __name__ == '__main__':
    print __doc__
#     print 'Shogun: ', bench_shogun()
    print 'scikits.learn: ', bench_skl()
#    print 'pybrain: ', bench_pybrain()
#    print 'MLPy: ', bench_mlpy()
    print 'MDP: ', bench_mdp()
