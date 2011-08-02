==========
Benchmarks
==========


We compare computation time for a few algorithms implemented in the
major machine learning toolkits accessible in Python.

**Last Update: July-2011**

We report the execution time for various machine learning libraries
exposed in Python: `MLPy <http://mlpy.fbk.eu/>`_, `PyBrain
<http://pybrain.org/>`_, `PyMVPA <http://pymvpa.org>`_, `MDP
<http://mdp-toolkit.sourceforge.net/>`_, `Shogun
<http://shogun-toolbox.org>`_ and `MiLK
<http://luispedro.org/software/milk>`_. Code for running the benchmarks
can be retrieved `its github repository
<http://github.com/scikit-learn>`_.

We also print the score on a validation dataset for all algorithms. For
classification algorithms, it's the fraction of correctly classified samples,
for regression algorithms it's the mean squared error and for k-means it's the
inertia criterion.


Software used
-------------

We used the latest released version as of June 2011:

  - scikit-learn 0.8
  - MDP 3.1
  - MLPy 2.2.2
  - PyMVPA 0.6.0~rc3
  - Shogun 0.10.0
  - Milk 0.3.9

I ran it on an Intel Core2 CPU @ 1.86GHz.


Datasets used
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
challenge. All details about the preparation of the data are found in the
technical report: *Design of experiments for the NIPS 2003 variable selection
benchmark*, Isabelle Guyon, July 2003.


Support Vector Machines
-----------------------

We used several Support Vector Machine (RBF kernel) implementations. Numbers
represent the time in seconds (lower is better) it took to train the dataset
and perform prediction on a test dataset. In the plot, results are normalized
to have the fastest method at 1.0.


.. table:: **Timing results in seconds**

   ============    =======     ======     ======     =======     ========    =============      ========
        Dataset     PyMVPA     Shogun        MDP     Pybrain         MLPy     scikit-learn          Milk
   ============    =======     ======     ======     =======     ========    =============      ========
        Madelon      11.52       5.63      40.48        17.5         9.47         **5.20**          5.76
        Arcene        1.30       0.39       4.87          --         1.61             0.38      **0.33**
   ============    =======     ======     ======     =======     ========    =============      ========


.. figure:: bench_svm.png
   :scale: 60%
   :align: center

The score by these classfifiers in in a test dataset is.


.. table:: **Classification score** - smaller is better

     ============    =======    ======    ====    =======   ===========   =============    ========
          Dataset     PyMVPA    Shogun    MDP     Pybrain          MLPy    scikit-learn        milk
     ============    =======    ======    ====    =======   ===========   =============    ========
          Madelon       0.65      0.65      --         --          0.65            0.65        0.50
          Arcene        0.73      0.73    0.73         --          0.73            0.73        0.73
     ============    =======    ======    ====    =======   ===========   =============    ========


.. warning::

     This is just meant as a sanity check, should not be taken at face
     value since parameters are not cross-validated, etc.


Nearest neighbors
-----------------

.. table:: **Timing results in seconds**

     ============      ========   ======     ====    ========    =============   ======
          Dataset        PyMVPA   Shogun      MDP        MLPy     scikit-learn    milk
     ============      ========   ======     ====    ========    =============   ======
          Madelon      **0.56**     1.36     0.58        1.41             0.57     8.24
          Arcene           0.10     0.22     0.10        0.21         **0.09**     1.33
     ============      ========   ======     ====    ========    =============   ======


.. figure:: bench_knn.png
   :scale: 60%
   :align: center


.. table:: **Classification score** - larger is better

     ============    =======    ======    ======   =========   =============  =====
          Dataset     PyMVPA    Shogun      MDP         MLPy    scikit-learn   milk
     ============    =======    ======    ======   =========   =============  =====
          Madelon       0.73      0.73      0.73        0.73            0.73   0.73
          Arcene        0.73      0.73      0.73        0.73            0.73   0.73
     ============    =======    ======    ======   =========   =============  =====


.. Logistic Regression
.. -------------------
..
.. TODO


K-means
-------

We run the k-means algorithm on both Madelon and Arcene dataset. To make sure
the methods are converging, we show in the second table the inertia of all
methods, which are mostly equivalent.

Note: The shogun is failling ..

.. table:: **Timing results in seconds**

     ============  =====   ========    =======   ========    =============    ========
          Dataset    MDP     Shogun    Pybrain       MLPy     scikit-learn        milk
     ============  =====   ========    =======   ========    =============    ========
          Madelon  35.75       0.68         NC       0.79             1.34    **0.67**
           Arcene   2.07   **0.19**      20.50       0.33             0.51        0.23
     ============  =====   ========    =======   ========    =============    ========


NC = Not Converging after one hour iteration.

.. figure:: bench_kmeans.png
   :scale: 60%
   :align: center


The following table shows the inertia, criterion that the k-means algorithm minimizes.

.. table:: **Inertia** - smaller is better

     ============   ==========  ========  ========     ===========    =============     ==============
          Dataset          MDP    Shogun   Pybrain            MLPy     scikit-learn               Milk
     ============   ==========  ========  ========     ===========    =============     ==============
          Madelon     7.4x10^8  7.3x10^8       --        7.3x10^8         7.4x10^8           7.3x10^8
           Arcene     1.4x10^9                 oo        1.4x10^9         1.4x10^9           1.4x10^9
     ============   ==========  ========  ========     ===========    =============     ==============


Elastic Net
-----------

We solve the elastic net using a coordinate descent algorithm on both Madelon and Arcene dataset.


.. table:: **Timing results in seconds**

     ============     =======    ========    =============
          Dataset      PyMVPA        MLPy     scikit-learn
     ============     =======    ========    =============
          Madelon        1.44        73.7         **0.52**
           Arcene        2.31       65.48         **1.90**
     ============     =======    ========    =============


.. figure:: bench_elasticnet.png
   :scale: 60%
   :align: center

.. table:: **Mean squared error** - smaller is better

     ============     =======    ========    =============
          Dataset     PyMVPA         MLPy     scikit-learn
     ============     =======    ========    =============
          Madelon       699.1      3759.8            597.1
           Arcene       84.92      151.28            65.39
     ============     =======    ========    =============


Lasso (LARS algorithm)
----------------------

We solve the Lasso model by Least Angle Regression (LARS) algorithm. MLPy and
scikit-learn use a pure Python implementation, while PyMVPA uses bindings to
R code.

We also show the Mean Squared error as a sanity check for the model. Note
that some NaN arise due to collinearity in the data.


.. table:: **Timing results in seconds**

     ============    =======  =========    =============
          Dataset     PyMVPA       MLPy     scikit-learn
     ============    =======  =========    =============
          Madelon      37.35      105.3         **1.17**
           Arcene      11.53       3.82         **2.95**
     ============    =======  =========    =============

.. figure:: bench_lars.png
   :scale: 60%
   :align: center


.. table:: **Mean Squared Error on a test dataset** - smaller is better

     ============  =======  =============    =============
          Dataset   PyMVPA           MLPy     scikit-learn
     ============  =======  =============    =============
          Madelon    567.0         682.32           680.91
           Arcene     87.5            NaN            65.39
     ============  =======  =============    =============



Principal Component Analysis
----------------------------

We run principal component analysis on the madelon datasets. In the libraries
that support it (scikit-learn, MDP, PyMVPA), we number of components in the
projection to 9. For the arcene dataset, most libraries could not handle the
memory requirements.


.. table:: **Timing results in seconds**

     ============     =======   ====   =======   =============   ========
          Dataset      PyMVPA    MDP   Pybrain    scikit-learn       milk
     ============     =======   ====   =======   =============   ========
          Madelon        0.48   0.47      8.93        **0.18**       3.07
     ============     =======   ====   =======   =============   ========


.. figure:: bench_pca.png
   :scale: 60%
   :align: center

.. table:: **Explained variance** - larger is better

     ============    =========   ========   ========   =============   =========
          Dataset       PyMVPA        MDP    Pybrain    scikit-learn       milk
     ============    =========   ========   ========   =============   =========
          Madelon     136705.5   136705.5   113545.8        135788.2    139158.7
     ============    =========   ========   ========   =============   =========



Misc
----

Author : Fabian Pedregosa

License : Simplified BSD
