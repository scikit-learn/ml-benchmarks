"""SVM benchmarks"""

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


def bench_shogun():
#
#       .. Shogun ..
#
    from shogun.Classifier import LibSVM
    from shogun.Features import RealFeatures, Labels
    from shogun.Kernel import GaussianKernel
    start = datetime.now()
    feat = RealFeatures(X.T)
    feat_test = RealFeatures(T.T)
    labels = Labels(y.astype(np.float64))
    kernel = GaussianKernel(feat, feat, 1.)
    shogun_svm = LibSVM(1., kernel, labels)
    shogun_svm.train()
    shogun_svm.classify(feat_test).get_labels()
    return datetime.now() - start


def bench_mlpy():
#
#       .. MLPy ..
#
    from mlpy import LibSvm as mlpy_svm
    start = datetime.now()
    mlpy_clf = mlpy_svm(kernel_type='rbf', C=1.)
    mlpy_clf.learn(X, y.astype(np.float64))
    mlpy_clf.pred(T)
    return datetime.now() - start


def bench_skl():
#
#       .. scikits.learn ..
#
    from scikits.learn import svm as skl_svm
    start = datetime.now()
    clf = skl_svm.SVC(kernel='rbf', C=1.)
    clf.fit(X, y)
    clf.predict(X)
    return datetime.now() - start


def bench_pymvpa():
#
#       .. PyMVPA ..
#
    from mvpa.clfs import svm as mvpa_svm
    from mvpa.datasets import dataset_wizard
    tstart = datetime.now()
    data = dataset_wizard(X, y)
    clf = mvpa_svm.RbfCSVMC(C=1.)
    clf.train(data)
    clf.predict(X)
    return datetime.now() - tstart


def bench_pybrain():
#
#       .. PyBrain ..
#
#   local import, they require libsvm < 2.81
    from pybrain.supervised.trainers.svmtrainer import SVMTrainer
    from pybrain.structure.modules.svmunit import SVMUnit
    from pybrain.datasets import SupervisedDataSet

    tstart = datetime.now()
    ds = SupervisedDataSet(X.shape[1], 1)
    for i in range(X.shape[0]):
        ds.addSample(X[i], y[i])
    clf = SVMTrainer(SVMUnit(), ds)
    clf.train()
    for i in range(T.shape[0]):
        clf.svm.model.predict(T[i])
    return datetime.now() - tstart



def bench_mdp():
#
#       .. MDP ..
#
    from mdp.nodes import LibSVMClassifier as mdp_svm
    start = datetime.now()
    clf = mdp_svm(kernel='RBF')
    clf.train(X, y)
    clf.label(T)
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

    res_pymvpa = bench(bench_pymvpa)
    print 'PyMVPA: mean %s, std %s' % (
        np.mean(res_pymvpa), np.std(res_pymvpa))

    res_pybrain = bench(bench_pybrain)
    print 'Pybrain: mean %s, std %s' % (
        np.mean(res_pybrain), np.std(res_pybrain))
