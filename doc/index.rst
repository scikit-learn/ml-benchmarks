==========
Benchmarks
==========


We compare computation time for a few algorithms implemented in the
major machine learning toolkits accessible in Python.

**Last Update: June-2011**

Time in seconds on the Madelon dataset for various machine learning libraries
exposed in Python: `MLPy <http://mlpy.fbk.eu/>`_, `PyBrain
<http://pybrain.org/>`_, `PyMVPA <http://pymvpa.org>`_, `MDP
<http://mdp-toolkit.sourceforge.net/>`_, `Shogun <http://shogun-toolbox.org>`_
and `MiLK <http://luispedro.org/software/milk>`_. Code for running the
benchmarks can be retrieved from http://github.com/scikit-learn.

We also plot the score on a validation dataset for all algorithms. For
classification algorithms, it's the fraction of correctly classified samples,
for regression algorithms it's the mean squared error and for k-means it's the
inertia criterion.


Used software
-------------

We used the latest released version as of June 2011:

  - scikits.learn 0.8
  - MDP 3.1
  - MLPy 2.2.2
  - PyMVPA 0.6.0.dev
  - Shogun 0.10.0

I ran it on an Intel(R) Core(TM)2 CPU @ 1.86GHz.


Used datasets
-------------


We use the Madelong and Arcene data set. The `Madelon data set
<http://archive.ics.uci.edu/ml/datasets/Madelon>`_, 4400 instances and 500
attributes, is an artificial dataset, which was part of the NIPS 2003
feature selection challenge. This is a two-class classification problem with
continuous input variables. The difficulty is that the problem is multivariate
and highly non-linear.

The `arcene data set <http://archive.ics.uci.edu/ml/datasets/Arcene>`_ task is
to distinguish cancer versus normal patterns from mass-spectrometric data.
This is a two-class classification problem with continuous input variables.
This dataset is one of 5 datasets of the NIPS 2003 feature selection
challenge. All details about the preparation of the data are found in our
technical report: Design of experiments for the NIPS 2003 variable selection
benchmark, Isabelle Guyon, July 2003.


Support Vector Machines
-----------------------

We used several Support Vector Machine (RBF kernel) implementations. Numbers
represent the time in seconds (lower is better) it took to train the dataset
and perform prediction on a test dataset. In the plot, results are normalized
so that the fastest method has value 1.0.



.. table:: Results in scikits.learn ml-benchmarks

     ============         =======           ======     ====     =======         ========    =============         ========
          Dataset          PyMVPA           Shogun      MDP     Pybrain             MLPy    scikits.learn             Milk
     ============         =======           ======     ====     =======         ========    =============         ========
          Madelon           12.89             6.03       --          --            10.88             6.23             4.90
          Arcene             1.37             0.42       --          --             1.75             0.41         **0.34**
     ============         =======           ======     ====     =======         ========    =============         ========



.. figure:: bench_svm.png
   :scale: 60%
   :align: center


The score by these calssfifiers in in a test dataset is.

.. warning::

     This is just meant as a sanity check, should not be taken at face
     value since parameters are not cross-validated, etc.

.. table:: Score in scikits.learn ml-benchmarks

     ============         =======           ======    ====      =======         ===========       =============         ========
          Dataset          PyMVPA           Shogun    MDP       Pybrain                MLPy       scikits.learn             milk
     ============         =======           ======    ====      =======         ===========       =============         ========
          Madelon             0.5              0.0      --           --                0.65                0.65              0.0
          Arcene             0.56             0.56      --           --                0.56                0.56             0.56
     ============         =======           ======    ====      =======         ===========       =============         ========



K-means
-------

bla bla bla. NC = not converging.

.. table:: Results in scikits.learn ml-benchmarks

     ============         =======       ======     ====     =======         ========    =============         ========
          Dataset         PyMVPA        Shogun      MDP     Pybrain             MLPy    scikits.learn             milk
     ============         =======       ======     ====     =======         ========    =============         ========
          Madelon              --           --     28.9          NC             0.79             1.36             0.55
           Arcene              --           --       --          --             0.81             1.77          **1.0**
     ============         =======       ======     ====     =======         ========    =============         ========


.. figure:: bench_kmeans.png
   :scale: 60%
   :align: center


The following table shows the inertia, criterion that the k-means algorithm minimizes.

.. table:: Inertia

     ============         =======           ======     =============     =======     =============    =============     ==============
          Inertia          PyMVPA           Shogun               MDP     Pybrain              MLPy    scikits.learn               Milk
     ============         =======           ======     =============     =======     =============    =============     ==============
          Madelon              --               --                --          --       739171883.6      745421891.3                 --
           Arcene              --               --     1403820558.52          --     1429740165.89      745421891.3      1451970835.28
     ============         =======           ======     =============     =======     =============    =============     ==============


Elastic Net
-----------

Bla bla bla bla


.. table:: Results in scikits.learn ml-benchmarks

     ============         =======    ========    =============
          Dataset         PyMVPA         MLPy    scikits.learn
     ============         =======    ========    =============
          Madelon            1.52        76.7             0.40
           Arcene            2.28        xxxx             1.90
     ============         =======    ========    =============


.. figure:: bench_svm.png
   :scale: 60%
   :align: center


Lasso (LARS algorithm)
----------------------


     ============         =======  =============     =======     =============    =============     ==============
              MSE          PyMVPA            MDP     Pybrain              MLPy    scikits.learn               Milk
     ============         =======  =============     =======     =============    =============     ==============
          Madelon              --             --          --       739171883.6      745421891.3                 --
           Arcene              --  1403820558.52          --     1429740165.89      745421891.3      1451970835.28
     ============         =======  =============     =======     =============    =============     ==============

TODO