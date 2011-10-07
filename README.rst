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

Results
----------

The latest maintained results of these benchmarks can be found on
http://scikit-learn.github.com/ml-benchmarks/

Others results of running these benchmarks on different boxes and with different software versions can 
be found on:

  - http://scikit-learn.sourceforge.net/ml-benchmarks/
  - http://fseoane.net/ml-benchmarks/
  - http://packages.python.org/milk/benchmarks.html

They differ because they are run with different versions of the packages, and different
compilation settings (e.g. linear algebra packs).

References
----------

  - scikit-learn : http://scikit-learn.sourceforge.net
  - MDP : http://mdp-toolkit.sourceforge.net/
  - PyMVPA : http://pymvpa.org
  - MLPy : https://mlpy.fbk.eu/
  - Shogun: http://www.shogun-toolbox.org/
  - PyBrain : http://pybrain.org/
  - Milk : http://luispedro.org/software/milk
  - Orange : http://orange.biolab.si/

Misc
----

Author: Fabian Pedregosa <fabian.pedregosa@inria.fr>

License: Simplified BSD
