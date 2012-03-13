"""Various libraries classifying on k-Nearest Neighbors"""

#
#       .. Imports ..
#
import numpy as np
from datetime import datetime

n_neighbors = 9


def bench_shogun(X, y, T, valid):
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
    score = np.mean(
        knn.classify(test_feat).get_labels()
        == valid)
    return score, datetime.now() - start


def bench_mdp(X, y, T, valid):
#
#       .. MDP ..
#
    from mdp.nodes.classifier_nodes import KNNClassifier
    start = datetime.now()
    knn_mdp = KNNClassifier(k=n_neighbors)
    knn_mdp.train(X, y)
    score = np.mean(knn_mdp.label(T) == valid)
    return score, datetime.now() - start


def bench_skl(X, y, T, valid):
#
#       .. scikits.learn ..
#
    from sklearn import neighbors
    start = datetime.now()
    clf = neighbors.NeighborsClassifier(n_neighbors=n_neighbors, algorithm='brute_inplace')
    clf.fit(X, y)
    score = np.mean(clf.predict(T) == valid)
    return score, datetime.now() - start


def bench_mlpy(X, y, T, valid):
#
#       .. MLPy ..
#
    from mlpy import Knn as mlpy_Knn
    start = datetime.now()
    mlpy_clf = mlpy_Knn(n_neighbors)
    mlpy_clf.compute(X, y)
    score = np.mean(mlpy_clf.predict(T) == valid)
    return score, datetime.now() - start


def bench_pymvpa(X, y, T, valid):
#
#       .. PyMVPA ..
#
    from mvpa.datasets import dataset_wizard
    from mvpa.clfs import knn as mvpa_knn
    start = datetime.now()
    data = dataset_wizard(X, y)
    mvpa_clf = mvpa_knn.kNN(k=n_neighbors)
    mvpa_clf.train(data)
    score = np.mean(mvpa_clf.predict(T) == valid)
    return score, datetime.now() - start


def bench_milk(X, y, T, valid):
#
#       .. milk ..
#
    from milk.supervised.knn import kNN
    start = datetime.now()
    learner = kNN(n_neighbors)
    model = learner.train(X, y)
    score = np.mean(map(model.apply, T) == valid)
    return score, datetime.now() - start


if __name__ == '__main__':
    import sys, misc

    # don't bother me with warnings
    import warnings; warnings.simplefilter('ignore')
    np.seterr(all='ignore')

    print __doc__ + '\n'
    if not len(sys.argv) == 2:
        print misc.USAGE % __file__
        sys.exit(-1)
    else:
        dataset = sys.argv[1]

    print 'Loading data ...'
    data = misc.load_data(dataset)

    print 'Done, %s samples with %s features loaded into ' \
      'memory' % data[0].shape

    score, res_shogun = misc.bench(bench_shogun, data)
    print 'Shogun: mean %.2f, std %.2f\n' % (res_shogun.mean(), res_shogun.std())
    print 'Score: %.2f' % score

    score, res_mdp = misc.bench(bench_mdp, data)
    print 'MDP: mean %.2f, std %.2f\n' % (res_mdp.mean(), res_mdp.std())
    print 'Score: %.2f' % score

    score, res_skl = misc.bench(bench_skl, data)
    print 'scikits.learn: mean %.2f, std %.2f\n' % (res_skl.mean(), res_skl.std())
    print 'Score: %.2f' % score

    score, res_mlpy = misc.bench(bench_mlpy, data)
    print 'MLPy: mean %.2f, std %.2f\n' % (res_mlpy.mean(), res_mlpy.std())
    print 'Score: %.2f' % score

    score, res_milk = misc.bench(bench_milk, data)
    print 'milk: mean %.2f, std %.2f\n' % (res_milk.mean(), res_milk.std())
    print 'Score: %.2f' % score

    score, res_pymvpa = misc.bench(bench_pymvpa, data)
    print 'PyMVPA: mean %.2f, std %.2f\n' % (res_pymvpa.mean(), res_pymvpa.std())
    print 'Score: %.2f' % score
