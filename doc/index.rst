==========
Benchmarks
==========


We compare computation time for a few algorithms implemented in the
major machine learning toolkits accessible in Python.


**Last Update: June-2011**

Time in seconds on the Madelon dataset for various machine learning libraries exposed in Python:
`MLPy <https://mlpy.fbk.eu/>`_, PyBrain, pymvpa, MDP, Shogun and MiLK. Code for running the
benchmarks can be retrieved from http://github.com/scikit-learn.


Madelon dataset
----------------

We use the Madelon data set \citep{Guyon2004}, 4400 instances and 500
attributes, that can be used in supervised and unsupervised settings and is
quite large, but small enough for most algorithms to run.

I ran it on my Intel Core2 6600, 2.40GHz CPU.

.. table:: Results in scikits.learn ml-benchmarks

     ============         =======           ======          =======         ========    =============         ========
        Benchmark          PyMVPA           Shogun          Pybrain             MLPy    scikits.learn             milk
     ============         =======           ======          =======         ========    =============         ========
             knn          **1.0**             2.23               --             2.23             3.05             2.20
      elasticnet               --               --               --           174.43          **1.0**               --
       lassolars               --               --               --            61.67          **1.0**               --
             pca               --               --               --               --          **1.0**            11.11
          kmeans               --             2.02          7057.02             1.61             6.74          **1.0**
             svm             3.35             1.20               --               --             1.24          **1.0**
     ============         =======           ======          =======         ========    =============         ========


All of the results are normalised by the fastest system for each entry (which
is therefore, by definition, 1.0).


Arcene dataset
--------------

========================       ====================
Data Set Characteristics
========================       ====================
Number of Instances                        900
Number of Attributes                     10000
Associated Tasks                Classification
========================       ====================

ARCENE's task is to distinguish cancer versus normal patterns from
mass-spectrometric data. This is a two-class classification problem with
continuous input variables. This dataset is one of 5 datasets of the NIPS 2003
feature selection challenge. All details about the preparation of the data are found in our technical
report: Design of experiments for the NIPS 2003 variable selection benchmark,
Isabelle Guyon, July 2003, [Web Link].




.. table:: Results in scikits.learn ml-benchmarks

     ============         =======           ======    =====          =======         ========    =============         ========
        Benchmark          PyMVPA           Shogun     MDP           Pybrain             MLPy    scikits.learn             milk
     ============         =======           ======    =====          =======         ========    =============         ========
             svm             1.37             0.44                        --             1.75             0.41         **0.36**
             knn             1.10             0.22     0.10               --             0.21         **0.09**           1.33
     ============         =======           ======    =====          =======         ========    =============         ========


The following table shows the classification score on a validation dataset for
classification algorithms.

.. warning::

     This is just meant as a sanity check, should not be taken at face
     value since parameters are not cross-validated, etc.


.. table:: Score in scikits.learn ml-benchmarks

     ============         =======           ======    ====      =======         ========    =============         ========
            Score          PyMVPA           Shogun    MDP       Pybrain             MLPy    scikits.learn             milk
     ============         =======           ======    ====      =======         ========    =============         ========
             svm             0.56              0.0                  --             0.56             0.56              0.0
             knn             0.73             0.73    0.73          --             0.73             0.73             0.73
     ============         =======           ======    ====      =======         ========    =============         ========

