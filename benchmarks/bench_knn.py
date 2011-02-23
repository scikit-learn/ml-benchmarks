"""Various libraries classifying on k-Nearest Neighbors"""

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
n_neighbors = 9


def bench_shogun():
#
#       .. Shogun ..
#
    from shogun import Classifier, Features, Distance
    start = datetime.now()
    feat = Features.RealFeatures(X.T)
    distance = Distance.EuclidianDistance(feat, feat)
    labels = Features.Labels(y.astype(np.float64))
    test_feat = Features.RealFeatures(T.T)
    knn = Classifier.KNN(n_neighbors, distance, labels)
    knn.train()
    knn.classify(test_feat).get_labels()
    return datetime.now() - start


def bench_mdp():
#
#       .. MDP ..
#
    from mdp.nodes.classifier_nodes import KNNClassifier
    start = datetime.now()
    knn_mdp = KNNClassifier(k=n_neighbors)
    knn_mdp.train(X, y)
    knn_mdp.label(T)
    return datetime.now() - start


def bench_skl():
#
#       .. scikits.learn ..
#
    from scikits.learn import neighbors
    start = datetime.now()
    clf = neighbors.NeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, y)
    clf.predict(T)
    return datetime.now() - start


def bench_mlpy():
#
#       .. MLPy ..
#
    from mlpy import Knn as mlpy_Knn
    start = datetime.now()
    mlpy_clf = mlpy_Knn(n_neighbors)
    mlpy_clf.compute(X, y)
    mlpy_clf.predict(T)
    return datetime.now() - start


def bench_pymvpa():
#
#       .. PyMVPA ..
#
    from mvpa.datasets import dataset_wizard
    from mvpa.clfs import knn as mvpa_knn
    start = datetime.now()
    data = dataset_wizard(X, y)
    mvpa_clf = mvpa_knn.kNN(k=n_neighbors)
    mvpa_clf.train(data)
    mvpa_clf.predict(T)
    return datetime.now() - start


if __name__ == '__main__':

    # don't bother me with warnings
    import warnings; warnings.simplefilter('ignore')
    np.seterr(all='ignore')

    print __doc__ + '\n'

    res_shogun = bench(bench_shogun)
    print 'Shogun: mean %s, std %s' % (res_shogun.mean(), res_shogun.std())

    res_mdp = bench(bench_mdp)
    print 'MDP: mean %s, std %s' % (res_mdp.mean(), res_mdp.std())

    res_skl = bench(bench_skl)
    print 'scikits.learn: mean %s, std %s' % (res_skl.mean(), res_skl.std())

    res_mlpy = bench(bench_mlpy)
    print 'MLPy: mean %s, std %s' % (res_mlpy.mean(), res_mlpy.std())

    res_pymvpa = bench(bench_pymvpa)
    print 'PyMVPA: mean %s, std %s' % (res_pymvpa.mean(), res_pymvpa.std())
