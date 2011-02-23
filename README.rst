
Benchmarks for various machine learning packages
==================================================

Collection of benchmarks comparing various python-based machine
learning packages.

This is meant to work with the development version of the libraries
scikits.learn, mlpy, pybrain, pymvpa, mdp and shogun. It might be hard
to get all packages working on the same machine, but benchmarks are
designed so that if something fail it will just print the exception
and go to the next one.

To execute a benchmark, just type from the prompt::

    $ python benchmarks/bench_$name.py

and you will se as output the mean and std deviation for the timing of
running the benchmark 10 times with its extreme values removed.

References
----------

  - scikits.learn : http://scikit-learn.sourceforge.net
  - MDP : http://mdp-toolkit.sourceforge.net/
  - PyMVPA : http://pymvpa.org
  - MLPy : https://mlpy.fbk.eu/
  - Shogun: http://www.shogun-toolbox.org/
  - PyBrain : http://pybrain.org/


Misc
----

Author: Fabian Pedregosa <fabian.pedregosa@inria.fr>
License: Simplified BSD