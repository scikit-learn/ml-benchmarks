"""bench different LARS implementations"""

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


def bench_skl():
#
#       .. scikits.learn ..
#
    from scikits.learn import linear_model
    start = datetime.now()
    skl_clf = linear_model.LassoLARS(alpha=0.)
    skl_clf.fit(X, y, normalize=False)
    skl_clf.predict(X)
    return datetime.now() - start
    

def bench_mlpy():
#
#       .. MLPy ..
#
    from mlpy import Lasso as mlpy_lasso
    start = datetime.now()
    mlpy_clf = mlpy_lasso(m=X.shape[1])
    mlpy_clf.learn(X, y)
    mlpy_clf.pred(X)
    return datetime.now() - start


def bench_pymvpa():
#
#       .. PyMVPA ..
#

    from mvpa.datasets import dataset_wizard
    from mvpa.clfs import lars as mvpa_lars
    tstart = datetime.now()
    data = dataset_wizard(X, y)
    mvpa_clf = mvpa_lars.LARS()
    mvpa_clf.train(data)
#    BROKEN
#    mvpa_pred = mvpa_clf.predict(X)
    return (datetime.now() - tstart)
    
    


if __name__ == '__main__':

    # don't bother me with warnings
    import warnings; warnings.simplefilter('ignore')
    np.seterr(all='ignore')

    print __doc__ + '\n'

    res_skl = bench(bench_skl)
    print 'scikits.learn: mean %s, std %s' % (res_skl.mean(), res_skl.std())

    res_mlpy = bench(bench_mlpy)
    print 'MLPy: mean %s, std %s' % (res_mlpy.mean(), res_mlpy.std())

    res_pymvpa = bench(bench_pymvpa)
    print 'PyMVPA: mean %s, std %s' % (res_pymvpa.mean(), res_pymvpa.std())


