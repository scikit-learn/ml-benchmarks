# code for the SVM bar plot
import itertools
import pylab as pl
import numpy as np
# bench on madelon

color = itertools.cycle('bgrcmykbgrcmyk')

### time for SVM
##time_madelon = (12.89, 6.03, 10.88, 6.23, 4.90)
##time_arcene = (1.37, 0.42, 1.75, 0.41, 0.34)
##soft = ('PyMVPA', 'Shogun', 'MLPy', 'scikits.learn', 'Milk')
##
### time for kmeans
##time_madelon = (6.03, 0.79, 1.36, 0.55)
##time_arcene = (1.86, 0.33, 0.55, 0.24)
##soft = ('MDP', 'MLPy', 'scikits.learn', 'Milk')

### LARS
##time_madelon = (36.32, 105.3, 1.17)
##time_arcene = (9.99, 3.82, 2.92)
##soft = ('PyMVPA', 'MLPy', 'scikits.learn')

### Elastic net
##time_madelon = (1.52, 76.7, 0.47)
##time_arcene  = (2.28, 65.48, 1.90)
##soft = ('PyMVPA', 'MLPy', 'scikits.learn')

### PCA
##time_madelon = (.48, .47, 8.93, .18, 3.07)
##time_arcene  = [0] * len(time_madelon)
##soft = ('PyMVPA', 'MDP', 'Pybrain', 'scikits.learn', 'milk')

# knn
time_madelon = (.56, 1.36, .58, 1.41, .57, 8.24)
time_arcene = (.10, .22, .1, .2, .09, 1.33)
soft = ('PyMVPA', 'Shogun', 'MDP', 'MLPy', 'scikits.learn', 'milk')

# normalize
time_madelon = time_madelon / np.min(time_madelon)
time_arcene = time_arcene / np.min(time_arcene)

pl.bar(np.arange(0, 3 * len(time_arcene), 3), time_madelon, color='b', label='Madelon dataset')
pl.bar(np.arange(0, 3 * len(time_arcene), 3) + 1, time_arcene, color='g', label='Arcene dataset')
pl.legend()
pl.xticks(np.arange(0, 3 * len(time_arcene), 3) + 1,
          soft)

pl.title('Time needed to perform train + predict (smaller is better)')
pl.show()

