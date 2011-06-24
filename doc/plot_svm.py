# code for the SVM bar plot
import itertools
import pylab as pl
import numpy as np
# bench on madelon

color = itertools.cycle('bgrcmykbgrcmyk')

### time for SVM
##time_madelon = (12.89, 6.03, 10.88, 6.23, 4.90)
##time_arcene = (1.37, 0.42, 1.75, 0.41, 0.34)
##soft = ('PyMVPA', 'Shogun', 'MLPy', 'scikits.learn', 'Milk'),
##
### time for kmeans
##time_madelon = (6.03, 0.79, 1.36, 0.55)
##time_arcene = (1.86, 0.33, 0.55, 0.24)
##soft = ('MDP', 'MLPy', 'scikits.learn', 'Milk'),

# LARS
time_madelon = (6.03, 0.79, 1.27, 0.55)
time_arcene = (1.86, 0.33, 0.55, 0.24)
soft = ('MDP', 'MLPy', 'scikits.learn', 'Milk'),


# normalize
time_madelon = time_madelon / np.min(time_madelon)
time_arcene = time_arcene / np.min(time_arcene)

pl.bar(np.arange(0, 3 * len(time_arcene), 3), time_madelon, color='b', label='Madelon dataset')
pl.bar(np.arange(0, 3 * len(time_arcene), 3) + 1, time_arcene, color='g', label='Arcene dataset')
pl.legend()
pl.xticks(np.arange(0, 3 * len(time_arcene), 3) + 1,
          *soft,
          rotation=0)
pl.title('Time needed to perform train + predict (smaller is better)')
pl.show()

