from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone, OutlierMixin, BaseEstimator
from sklearn.cluster import KMeans


class DividedOutlierDetector(BaseEstimator, OutlierMixin):
    """
    Outlier detector that divides the input dataset into clusters and applies outlier detection to each cluster
    separately. The resulting outliers are then combined and returned as the final set of outliers.

    Parameters:
    ----------
    detector: object
        The outlier detection algorithm that will be applied to each cluster.
    clusterer: object, default=KMeans(random_state=0)
        The clustering algorithm that will divide the input dataset into clusters.
    verbose: int, default=0
        Controls the verbosity of the algorithm.
    n_jobs: int, default=-1
        The number of parallel jobs to use for outlier detection.

    Attributes:
    ----------
    detectors_: list
        The outlier detection algorithms that were fit to each cluster.
    labels_: array-like
        The cluster labels for each sample in the input dataset.
    """

    _estimator_type = "outlier_detector"

    def __init__(
        self,
        detector: object,
        clusterer: object = KMeans(random_state=0),
        verbose: int = 0,
        n_jobs: int = -1,
    ):
        self.detector = detector
        self.clusterer = clusterer
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit_resample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit the outlier detection algorithm to each cluster in the input dataset.

        Parameters
        ----------
        X: pd.DataFrame
            The input data to fit the algorithm on.
        y: pd.Series
            The true labels for the input data.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            The outliers in the input dataset.
        """

        # Use the clusterer object to predict cluster labels for the input data X using y as true labels
        self.labels_ = self.clusterer.fit_predict(X, y)

        # Create a list of boolean masks, where each mask corresponds to a cluster label
        masks = [self.labels_ == label for label in set(self.labels_)]

        # Create a list of delayed job objects, where each job applies the detector object to the input data X
        # with a particular mask (corresponding to a particular cluster label)
        jobs = (delayed(od_path)(clone(self.detector), X, y, mask) for mask in masks)

        # Use the Parallel function to run the delayed jobs in parallel and get the paths of outliers and outliers
        # for each cluster
        paths = Parallel(
            backend="multiprocessing",
            max_nbytes="256M",
            pre_dispatch="all",
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )(jobs)

        # Unzip the paths to get the outliers and outliers for each cluster, as well as the corresponding detector
        # objects
        xx, yy, self.detectors_ = zip(*paths)

        # Concatenate the outliers for all clusters to get the final set of outliers
        X_in = pd.concat(xx)
        y_in = pd.concat(yy)

        # Return the final set of outliers
        return X_in, y_in


def od_path(
    detector: object, X: pd.DataFrame, y: pd.Series, ind: np.ndarray
) -> Tuple[pd.DataFrame, pd.Series, object]:
    """
    Perform outlier detection on a subset of the data and return a subset of the data and labels
    that do not contain outliers, along with the detector used to identify outliers.

    Parameters
    ----------
    detector : object
        Outlier detection algorithm
    X : pd.DataFrame
        Feature matrix of shape (n_samples, n_features)
    y : pd.Series
        Label vector of shape (n_samples,)
    ind : np.ndarray
        Indices of the subset of data to perform outlier detection on

    Returns
    -------
    tuple : Tuple[np.ndarray, np.ndarray, LocalOutlierFactor]
        Subset of X and y without outliers, and the detector used
    """

    # Subset the input data based on the provided indices
    X, y = X[ind], y[ind]

    # Fit the outlier detection algorithm to the subset of input data
    labels = detector.fit_predict(X)

    # Identify the outliers in the subset of input data
    out = labels < 0

    # Return a subset of X and y that do not contain outliers, along with the outlier detection algorithm used
    return X[~out], y[~out], detector
