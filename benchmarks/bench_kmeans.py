"""K-means clustering"""

import numpy as np
from datetime import datetime

#
#       .. Load dataset ..
#
from misc import load_data, bench
print 'Loading data ...'
X, y, T = load_data()
print 'Done, %s samples with %s features loaded into ' \
      'memory' % X.shape
n_components = 9


def bench_shogun():
#
#       .. Shogun ..
#
    from shogun.Distance import EuclidianDistance
    from shogun.Features import RealFeatures
    from shogun.Clustering import KMeans
    start = datetime.now()
    feat = RealFeatures(X.T)
    distance=EuclidianDistance(feat, feat)
    clf=KMeans(n_components, distance)
    clf.train()
    return datetime.now() - start


def bench_skl():
#
#       .. scikits.learn ..
#
    from scikits.learn import cluster as skl_cluster
    start = datetime.now()
    clf = skl_cluster.KMeans(k=n_components, n_init=1)
    clf.fit(X)
    return datetime.now() - start


def bench_pybrain():
#
#       .. pybrain ..
#
    from pybrain.auxiliary import kmeans as pybrain_kmeans
    start = datetime.now()
    pybrain_kmeans.kmeanspp(X, n_components)
    return datetime.now() - start


def bench_mlpy():
#
#       .. MLPy ..
#
    from mlpy import Kmeans as mlpy_Kmeans
    start = datetime.now()
    clf = mlpy_Kmeans(n_components)
    clf.compute(X)
    return datetime.now() - start


def bench_mdp():
#
#       .. MDP ..
#
    from mdp.nodes import KMeansClassifier as mdp_KMeans
    start = datetime.now()
    clf = mdp_KMeans(n_components)
    clf.label(X)
    return datetime.now() - start



if __name__ == '__main__':

    # don't bother me with warnings
    import warnings; warnings.simplefilter('ignore')
    np.seterr(all='ignore')

    print __doc__ + '\n'


    res_shogun = bench(bench_shogun)
    print 'Shogun: mean %s, std %s' % (
        np.mean(res_shogun), np.std(res_shogun))

    res_mdp = bench(bench_mdp)
    print 'MDP: mean %s, std %s' % (
        np.mean(res_mdp), np.std(res_mdp))

    res_skl = bench(bench_skl)
    print 'scikits.learn: mean %s, std %s' % (
        np.mean(res_skl), np.std(res_skl))

    res_mlpy = bench(bench_mlpy)
    print 'MLPy: mean %s, std %s' % (
        np.mean(res_mlpy), np.std(res_mlpy))

    res_pybrain = bench(bench_pybrain)
    print 'Pybrain: mean %s, std %s' % (
        np.mean(res_pybrain), np.std(res_pybrain))
