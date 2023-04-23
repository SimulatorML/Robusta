from typing import Union, Callable

import pandas as pd
from sklearn.base import clone, BaseEstimator

from . import OutlierDetector


class SupervisedOutlierDetector(OutlierDetector):
    """
    A supervised outlier detector that uses a scoring function to assign outlier labels based on a user-specified threshold.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator that implements `fit` and `predict` methods.
    scoring : callable
        A function that takes the true labels and predicted labels as input and returns a score for each sample.
    thresh : float
        The threshold value used to assign outlier labels. Samples with scores below this value are considered outliers.

    Attributes
    ----------
    estimator_ : object
        The fitted estimator.

    Methods
    -------
    fit_predict(X, y)
        Fits the estimator to the data and returns the outlier labels.

    """
    _estimator_type = 'outlier_detector'

    def __init__(self,
                 estimator: BaseEstimator,
                 scoring: Union[Callable, str],
                 thresh: float):
        self.estimator_ = None
        self.estimator = estimator
        self.scoring = scoring  # by sample
        self.thresh = thresh

    def fit_predict(self,
                    X: pd.DataFrame,
                    y: pd.Series) -> pd.Series:
        """
        Fit the estimator to the data and predict outlier labels.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input data.
        y : pandas Series of shape (n_samples,)
            The true labels.

        Returns
        -------
        labels : pandas Series of shape (n_samples,)
            The predicted outlier labels. Outliers are assigned a label of -1, and outliers are assigned a label of 1.

        """

        # clone the estimator object to avoid modifying the original object
        self.estimator_ = clone(self.estimator).fit(X, y)

        # predict the labels of the input data using the fitted estimator
        y_pred = self.estimator_.predict(X)

        # calculate the outlier scores based on the difference between true and predicted labels
        scores = self.scoring(y, y_pred)

        # create a series of scores with the same index as the input labels
        scores = pd.Series(scores, index=y.index)

        # create a series of labels with the same index as the input labels, initialized as None
        labels = pd.Series(None, index=y.index)

        # assign a label of -1 to data points with scores below the threshold, and 1 otherwise
        outliers = scores < self.thresh
        labels[outliers] = -1
        labels[~outliers] = 1

        return labels
