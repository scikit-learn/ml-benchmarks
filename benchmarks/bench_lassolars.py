"""bench different LARS implementations"""

import numpy as np
from datetime import datetime


def bench_skl(X, y, T, valid):
#
#       .. scikits.learn ..
#
    from scikits.learn import linear_model
    start = datetime.now()
    skl_clf = linear_model.LassoLARS(alpha=0.)
    skl_clf.fit(X, y, normalize=False)
    pred = skl_clf.predict(T)
    delta = datetime.now() - start
    mse = np.linalg.norm(pred - valid, 2)**2
    return mse, delta


def bench_mlpy(X, y, T, valid):
#
#       .. MLPy ..
#
    from mlpy import Lasso as mlpy_lasso
    start = datetime.now()
    mlpy_clf = mlpy_lasso(m=X.shape[1])
    mlpy_clf.learn(X, y)
    pred = mlpy_clf.pred(T)
    import pdb; pdb.set_trace()
    delta = datetime.now() - start
    mse = np.linalg.norm(pred - valid, 2)**2
    return mse, delta


def bench_pymvpa(X, y, T, valid):
#
#       .. PyMVPA ..
#

    from mvpa.datasets import dataset_wizard
    from mvpa.clfs import lars as mvpa_lars
    tstart = datetime.now()
    data = dataset_wizard(X, y)
    mvpa_clf = mvpa_lars.LARS()
    mvpa_clf.train(data)
    delta  = datetime.now() - start
#    BROKEN
#    mvpa_pred = mvpa_clf.predict(X)
    return None, delta


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

##    score, res_skl = misc.bench(bench_skl, data)
##    print 'scikits.learn: mean %.2f, std %.2f' % (
##        np.mean(res_skl), np.std(res_skl))
##    print 'MSE: %s\n' % score

    score, res_mlpy = misc.bench(bench_mlpy, data)
    print 'MLPy: mean %.2f, std %.2f' % (
        np.mean(res_mlpy), np.std(res_mlpy))
    print 'MSE: %s\n' % score

    score, res_pymvpa = misc.bench(bench_pymvpa, data)
    print 'PyMVPA: mean %.2f, std %.2f' % (
        np.mean(res_pymvpa), np.std(res_pymvpa))
    print 'MSE: %s\n' % score
