"""PCA benchmarks"""

import numpy as np
from datetime import datetime

n_components = 9

def explained_variance(X, W):
    """
    We compute explained variance from the principal directions W using the
    principle that W are the eigenvectors for the covariance matrix dot(X.T,
    X).
    """
    mean = np.mean(X, axis=0)
    X -= mean
    C = np.dot(X.T, X)
    s = np.zeros(W.shape[0])
    for i in range(W.shape[0]):
        s[i] = np.dot(np.dot(W[i], C.T), W[i].T) / np.dot(W[i], W[i].T)
    return s / X.shape[0]


def bench_skl(X, y, T, valid):
#
#       .. scikits.learn ..
#
    from scikits.learn import decomposition
    start = datetime.now()
    clf = decomposition.RandomizedPCA(n_components=n_components)
    clf.fit(X)
    ev = explained_variance(X, clf.components_).sum()
    return ev, datetime.now() - start


def bench_pybrain(X, y, T, valid):
#
#       .. pybrain ..
#
    from pybrain.auxiliary import pca
    start = datetime.now()
    W = pca.pPca(X, n_components)
    ev = explained_variance(X, W).sum()
    return ev, datetime.now() - start


def bench_mdp(X, y, T, valid):
#
#       .. MDP ..
#
    from mdp.nodes import PCANode
    start = datetime.now()
    clf = PCANode(output_dim=n_components)
    clf.train(X)
    clf.stop_training()
    ev = explained_variance(X, clf.v.T).sum()
    return ev, datetime.now() - start


def bench_pymvpa(X, y, T, valid):
#
#       .. PyMVPA ..
#
    from mvpa.mappers.mdp_adaptor import PCAMapper as MVPA_PCA
    from mvpa.datasets import dataset_wizard
    start = datetime.now()
    clf = MVPA_PCA(output_dim=n_components)
    data = dataset_wizard(samples=X)
    clf.train(data)
    ev = explained_variance(X, clf.proj.T).sum()
    return ev, datetime.now() - start


def bench_milk(X, y, T, valid):
#
#       .. milk ..
#
    from milk.unsupervised import pca
    start = datetime.now()
    Y, W = pca(X, zscore=False)
    ev = explained_variance(X, W).sum()
    return ev, datetime.now() - start


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

    score, res_mdp = misc.bench(bench_mdp, data)
    print 'MDP: mean %s, std %s' % (
        np.mean(res_mdp), np.std(res_mdp))
    print 'Explained variance: %s\n'% score

    score, res_skl = misc.bench(bench_skl, data)
    print 'scikits.learn: mean %.2f, std %.2f' % (
        np.mean(res_skl), np.std(res_skl))
    print 'Explained variance: %s\n'% score

    score, res_pybrain = misc.bench(bench_pybrain, data)
    print 'Pybrain: mean %s, std %s' % (
        np.mean(res_pybrain), np.std(res_pybrain))
    print 'Explained variance: %s\n'% score

    score, res_milk = misc.bench(bench_milk, data)
    print 'milk: mean %s, std %s' % (
        np.mean(res_milk), np.std(res_milk))
    print 'Explained variance: %s\n'% score

    score, res_pymvpa = misc.bench(bench_pymvpa, data)
    print 'PyMVPA: mean %s, std %s' % (
        np.mean(res_pymvpa), np.std(res_pymvpa))
    print 'Explained variance: %s\n'% score

