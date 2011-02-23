"""
The glmnet.R script must be called before to generate the data
"""

import numpy as np
from datetime import datetime
from scikits.learn import linear_model


X = np.loadtxt('X.csv', skiprows=1, usecols=np.arange(1, 501))
y = np.loadtxt('y.csv', skiprows=1, usecols=(1,))

print 'X.shape: ', X.shape
print 'y.shape: ', y.shape

start = datetime.now()
clf = linear_model.lasso_path(X, y)
print datetime.now() - start

