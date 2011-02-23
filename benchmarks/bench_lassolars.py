"""bench different LARS implementations"""

from datetime import datetime
from scikits.learn import linear_model
from mlpy import Lasso as mlpy_lasso
from mvpa.datasets import Dataset
from mvpa.clfs import lars as mvpa_lars

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
    start = datetime.now()
    skl_clf = linear_model.LassoLARS(alpha=0.)
    skl_clf.fit(X, y, normalize=False)
    skl_clf.predict(X)
    return datetime.now() - start
    

def bench_mlpy():
#
#       .. MLPy ..
#
    start = datetime.now()
    mlpy_clf = mlpy_lasso(m=X.shape[1])
    mlpy_clf.learn(X, y)
    mlpy_clf.pred(X)
    return datetime.now() - start


def bench_pymvpa():
#
#       .. PyMVPA ..
#

    tstart = datetime.now()
    data = Dataset(samples=X, labels=y)
    mvpa_clf = mvpa_lars.LARS()
    mvpa_clf.train(data)
#    BROKEN
#    mvpa_pred = mvpa_clf.predict(X)
    return (datetime.now() - tstart)
    
    


if __name__ == '__main__':
    print __doc__ 
    print 'scikits.learn: ', bench(bench_skl)
    print 'MLPy: ', bench(bench_mlpy)
    print 'PyMVPA: ', bench(bench_pymvpa)
