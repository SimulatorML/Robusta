.. outliers:

########
Outliers
########

**Anomaly Detection** is a data analysis technique aimed at identifying unusual and anomalous values in a dataset.

.. image:: ../media/anomaly_detection.png
    :align: center
    :width: 100%
    :alt: plot of the anomaly detection
    
    
Divide Outlier Detector
=======================

**DividedOutlierDetector** is an algorithm that splits the data into several subgroups and determines the outliers in each of them. This method is based on the assumption that outliers tend to cluster together, making them more visible.

.. autoclass:: robusta.outliers.DividedOutlierDetector
    :members:
    
.. tip::

    The DividedOutlierDetector relies on statistical measures to identify potential outliers. It's important to choose a robust measure that is not overly sensitive to outliers itself. The median absolute deviation (MAD) is a good choice for this purpose.
    
.. caution::
    
    The effectiveness of DividedOutlierDetector relies heavily on the selection of subgroups. If the subgroups are not properly selected, this method may fail to identify outliers or even produce false positives. Careful consideration must be given to subgroup selection, which should be based on factors such as the nature of the data, the research question, and the intended use of the results.


Local Outlier Factor
====================

The **Local Outlier Factor** (LOF) is an unsupervised algorithm used for outlier detection in data sets. It measures the local density deviation of a given data point with respect to its neighbors. The basic idea behind LOF is that outliers will have a lower local density than their surrounding neighbors.

.. autoclass:: robusta.outliers.LocalOutlierFactor
    :members:

.. tip::

    Normalize the data to ensure that all features have a similar scale.
    
.. caution::

    LOF may not be well-suited for high-dimensional data sets, as the distance metrics used may not be appropriate, and the algorithm may become computationally expensive
