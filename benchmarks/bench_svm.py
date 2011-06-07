"""SVM benchmarks"""

import numpy as np
from datetime import datetime


def bench_shogun(X, y, T, valid):
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
    score = np.mean(
        shogun_svm.classify(feat_test).get_labels()
        == valid)
    return score, datetime.now() - start


def bench_mlpy(X, y, T, valid):
#
#       .. MLPy ..
#
    from mlpy import LibSvm as mlpy_svm
    start = datetime.now()
    mlpy_clf = mlpy_svm(kernel_type='rbf', C=1.)
    mlpy_clf.learn(X, y.astype(np.float64))
    score = np.mean(mlpy_clf.pred(T) == valid)
    return score, datetime.now() - start


def bench_skl(X, y, T, valid):
#
#       .. scikits.learn ..
#
    from scikits.learn import svm as skl_svm
    start = datetime.now()
    clf = skl_svm.SVC(kernel='rbf', C=1.)
    clf.fit(X, y)
    score = np.mean(clf.predict(T) == valid)
    return score, datetime.now() - start


def bench_pymvpa(X, y, T, valid):
#
#       .. PyMVPA ..
#
    from mvpa.clfs import svm as mvpa_svm
    from mvpa.datasets import dataset_wizard
    tstart = datetime.now()
    data = dataset_wizard(X, y)
    clf = mvpa_svm.RbfCSVMC(C=1.)
    clf.train(data)
    score = np.mean(clf.predict(T) == valid)
    return score, datetime.now() - tstart


def bench_pybrain(X, y, T, valid):
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
    pred = np.empty(T.shape[0], dtype=np.int32)
    for i in range(T.shape[0]):
        pred[i] = clf.svm.model.predict(T[i])
    score = np.mean(pred == valid)
    return score, datetime.now() - tstart



def bench_mdp(X, y, T, valid):
#
#       .. MDP ..
#
    from mdp.nodes import LibSVMClassifier as mdp_svm
    start = datetime.now()
    clf = mdp_svm(kernel='RBF')
    clf.train(X, y)
    score = np.mean(clf.label(T) == valid)
    return score, datetime.now() - start


def bench_milk(X, y, T, valid):
#
#       .. milk ..
#
    from milk.supervised import svm
    start = datetime.now()
    learner = svm.svm_raw(kernel=svm.rbf_kernel(sigma=1.), C=1.)
    model = learner.train(X,y)
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

    res_shogun = misc.bench(bench_shogun, data)
    print 'Shogun: mean %.2f, std %.2f\n' % (
        np.mean(res_shogun), np.std(res_shogun))

    res_mdp = misc.bench(bench_mdp, data)
    print 'MDP: mean %.2f, std %.2f\n' % (
        np.mean(res_mdp), np.std(res_mdp))

    res_skl = misc.bench(bench_skl, data)
    print 'scikits.learn: mean %.2f, std %.2f\n' % (
        np.mean(res_skl), np.std(res_skl))

    res_mlpy = misc.bench(bench_mlpy, data)
    print 'MLPy: mean %.2f, std %.2f\n' % (
        np.mean(res_mlpy), np.std(res_mlpy))

    res_pymvpa = misc.bench(bench_pymvpa, data)
    print 'PyMVPA: mean %.2f, std %.2f\n' % (
        np.mean(res_pymvpa), np.std(res_pymvpa))

    res_pybrain = misc.bench(bench_pybrain, data)
    print 'Pybrain: mean %.2f, std %.2f\n' % (
        np.mean(res_pybrain), np.std(res_pybrain))

    res_milk = misc.bench(bench_milk, data)
    print 'milk: mean %.2f, std %.2f\n' % (
        np.mean(res_milk), np.std(res_milk))
