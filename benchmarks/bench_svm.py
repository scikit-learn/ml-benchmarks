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
    kernel = GaussianKernel(feat, feat, sigma)
    shogun_svm = LibSVM(1., kernel, labels)
    shogun_svm.train()
    dec_func = shogun_svm.classify(feat_test).get_labels()
    score = np.mean(np.sign(dec_func) == valid)
    return score, datetime.now() - start


def bench_mlpy(X, y, T, valid):
#
#       .. MLPy ..
#
    from mlpy import LibSvm
    start = datetime.now()
    clf = LibSvm(kernel_type='rbf', C=1., gamma=1. / sigma)
    clf.learn(X, y.astype(np.float64))
    score = np.mean(clf.pred(T) == valid)
    return score, datetime.now() - start


def bench_skl(X, y, T, valid):
#
#       .. scikits.learn ..
#
    from sklearn import svm as skl_svm
    start = datetime.now()
    clf = skl_svm.SVC(kernel='rbf', C=1., gamma=1. / sigma)
    clf.fit(X, y)
    score = np.mean(clf.predict(T) == valid)
    return score, datetime.now() - start


def bench_pymvpa(X, y, T, valid):
#
#       .. PyMVPA ..
#
    from mvpa.clfs import svm
    from mvpa.datasets import dataset_wizard
    tstart = datetime.now()
    data = dataset_wizard(X, y)
    kernel = svm.RbfSVMKernel(gamma=1. / sigma)
    clf = svm.SVM(C=1., kernel=kernel)
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
    from mdp.nodes import LibSVMClassifier
    start = datetime.now()
    clf = LibSVMClassifier(kernel='RBF')
    clf.parameter.gamma = 1. / sigma
    clf.train(X, y)
    score = np.mean(clf.label(T) == valid)
    return score, datetime.now() - start


def bench_milk(X, y, T, valid):
#
#       .. milk ..
#
    from milk.supervised import svm
    start = datetime.now()
    learner = svm.svm_raw(
        kernel=svm.rbf_kernel(sigma=sigma), C=1.)
    model = learner.train(X, y)
    pred = np.sign(map(model.apply, T))
    score = np.mean(pred == valid)
    return score, datetime.now() - start


def bench_orange(X, y, T, valid):
#
#       .. Orange ..
#
    import orange
    start = datetime.now()

    # prepare data in Orange's format
    columns = []
    for i in range(0, X.shape[1]):
        columns.append("a" + str(i))
    [orange.EnumVariable(x) for x in columns]
    classValues = ['0', '1']

    domain = orange.Domain(map(orange.FloatVariable, columns),
                   orange.EnumVariable("class", values=classValues))
    y.shape = (len(y), 1) #reshape for Orange
    y[np.where(y < 0)] = 0 # change class labels to 0..K
    orng_train_data = orange.ExampleTable(domain, np.hstack((X, y)))

    valid.shape = (len(valid), 1)  #reshape for Orange
    valid[np.where(valid < 0)] = 0 # change class labels to 0..K
    orng_test_data = orange.ExampleTable(domain, np.hstack((T, valid)))

    learner = orange.SVMLearner(orng_train_data, \
                                svm_type=orange.SVMLearner.Nu_SVC, \
                                kernel_type=orange.SVMLearner.RBF, C=1., \
                                gamma=1. / sigma)

    pred = np.empty(T.shape[0], dtype=np.int32)
    for i, e in enumerate(orng_test_data):
        pred[i] = learner(e)

    score = np.mean(pred == valid)
    return score, datetime.now() - start


if __name__ == '__main__':
    import sys
    import misc

    # don't bother me with warnings
    import warnings
    warnings.simplefilter('ignore')
    np.seterr(all='ignore')

    print __doc__ + '\n'
    if not len(sys.argv) == 2:
        print misc.USAGE % __file__
        sys.exit(-1)
    else:
        dataset = sys.argv[1]

    print 'Loading data ...'
    data = misc.load_data(dataset)

    # set sigma to something useful
    from milk.unsupervised import pdist
    sigma = np.median(pdist(data[0]))

    print 'Done, %s samples with %s features loaded into ' \
      'memory' % data[0].shape

    score, res_shogun = misc.bench(bench_shogun, data)
    print 'Shogun: mean %.2f, std %.2f' % (
        np.mean(res_shogun), np.std(res_shogun))
    print 'Score: %.2f\n' % score

    score, res_mdp = misc.bench(bench_mdp, data)
    print 'MDP: mean %.2f, std %.2f' % (
        np.mean(res_mdp), np.std(res_mdp))
    print 'Score: %.2f\n' % score

    score, res_skl = misc.bench(bench_skl, data)
    print 'scikits.learn: mean %.2f, std %.2f' % (
        np.mean(res_skl), np.std(res_skl))
    print 'Score: %.2f\n' % score

    score, res_mlpy = misc.bench(bench_mlpy, data)
    print 'MLPy: mean %.2f, std %.2f' % (
        np.mean(res_mlpy), np.std(res_mlpy))
    print 'Score: %.2f\n' % score

    score, res_pymvpa = misc.bench(bench_pymvpa, data)
    print 'PyMVPA: mean %.2f, std %.2f' % (
        np.mean(res_pymvpa), np.std(res_pymvpa))
    print 'Score: %.2f\n' % score

    score, res_pybrain = misc.bench(bench_pybrain, data)
    print 'Pybrain: mean %.2f, std %.2f' % (
        np.mean(res_pybrain), np.std(res_pybrain))
    print 'Score: %.2f\n' % score

    score, res_milk = misc.bench(bench_milk, data)
    print 'milk: mean %.2f, std %.2f' % (
        np.mean(res_milk), np.std(res_milk))
    print 'Score: %.2f\n' % score

    score, res_orange = misc.bench(bench_orange, data)
    print 'Orange: mean %.2f, std %.2f' % (
        np.mean(res_orange), np.std(res_orange))
    print 'Score: %.2f\n' % score
