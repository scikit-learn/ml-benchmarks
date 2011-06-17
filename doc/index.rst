==========
Benchmarks
==========


We compare computation time for a few algorithms implemented in the
major machine learning toolkits accessible in Python.

**Last Update: June-2011**

Time in seconds on the Madelon dataset for various machine learning libraries
exposed in Python: `MLPy <https://mlpy.fbk.eu/>`_, `PyBrain
<http://pybrain.org/>`_, PyMVPA, MDP, Shogun and MiLK. Code for running the
benchmarks can be retrieved from http://github.com/scikit-learn.


Used software
-------------

We used the latest released version as of June 2011:

  - scikits.learn 0.8
  - MDP 3.1
  - MLPy 2.2.2
  - PyMVPA 0.6.0.dev
  - Shogun 0.10.0

I ran it on my Intel Core2 6600, 2.40GHz CPU.

Madelon dataset
----------------


========================       ====================
Data Set Characteristics
========================       ====================
Number of Instances                       4400
Number of Attributes                       500
Associated Tasks                Classification
========================       ====================


We use the `Madelon data set
<http://archive.ics.uci.edu/ml/datasets/Madelon>`_, 4400 instances and 500
attributes, is an artificial dataset, which was part of the NIPS 2003
feature selection challenge. This is a two-class classification problem with
continuous input variables. The difficulty is that the problem is multivariate
and highly non-linear.



.. table:: Results in scikits.learn ml-benchmarks

     ============         =======           ======     ====     =======         ========    =============         ========
        Benchmark          PyMVPA           Shogun      MDP     Pybrain             MLPy    scikits.learn             milk
     ============         =======           ======     ====     =======         ========    =============         ========
          kmeans               --               --       --          --             0.81             1.77          **1.0**
             svm            12.89             6.03       --          --            10.88             6.23             4.90
     ============         =======           ======     ====     =======         ========    =============         ========


The following table shows the classification score on a validation dataset.
For classification algorithms, it's the fraction of correctly classified
samples, for regression algorithms it's the mean squared error and for k-means it's the inertia criterion.

.. warning::

     This is just meant as a sanity check, should not be taken at face
     value since parameters are not cross-validated, etc.

.. table:: Score in scikits.learn ml-benchmarks

     ============         =======           ======    ====      =======         ===========       =============         ========
            Score          PyMVPA           Shogun    MDP       Pybrain                MLPy       scikits.learn             milk
     ============         =======           ======    ====      =======         ===========       =============         ========
             svm             0.5               0.0                   --                0.65             0.65              0.0
          kmeans               --             2.02      --      7057.02         739171883.6         745421891.3          **1.0**
     ============         =======           ======    ====      =======         ===========       =============         ========



Arcene dataset
--------------

========================       ====================
Data Set Characteristics
========================       ====================
Number of Instances                        900
Number of Attributes                     10000
Associated Tasks                Classification
========================       ====================

The `arcene data set <http://archive.ics.uci.edu/ml/datasets/Arcene>`_ task is
to distinguish cancer versus normal patterns from mass-spectrometric data.
This is a two-class classification problem with continuous input variables.
This dataset is one of 5 datasets of the NIPS 2003 feature selection
challenge. All details about the preparation of the data are found in our
technical report: Design of experiments for the NIPS 2003 variable selection
benchmark, Isabelle Guyon, July 2003.




.. table:: Results in scikits.learn ml-benchmarks

     ============         =======           ======    =====          =======         ========    =============         ========
        Benchmark          PyMVPA           Shogun     MDP           Pybrain             MLPy    scikits.learn             milk
     ============         =======           ======    =====          =======         ========    =============         ========
             svm             1.37             0.44                        --             1.75             0.41         **0.36**
             knn             1.10             0.22     0.10               --             0.21         **0.09**           1.33
       lassolars            10.99               --       --               --             3.80             1.16
     ============         =======           ======    =====          =======         ========    =============         ========




.. table:: Score in scikits.learn ml-benchmarks

     ============         =======           ======    ====      =======         ========    =============         ========
            Score          PyMVPA           Shogun    MDP       Pybrain             MLPy    scikits.learn             milk
     ============         =======           ======    ====      =======         ========    =============         ========
             svm             0.56              0.0                  --             0.56             0.56              0.0
             knn             0.73             0.73    0.73          --             0.73             0.73             0.73
        lassolars             NaN               --      --          --              NaN            61.94
     ============         =======           ======    ====      =======         ========    =============         ========


