from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone, ClassifierMixin, BaseEstimator
from sklearn.utils.metaestimators import _BaseComposition, if_delegate_has_method

from ..utils import logmsg


class PseudoLabeling(_BaseComposition, ClassifierMixin):
    """
    A class for implementing the Pseudo-Labeling algorithm for semi-supervised learning.

    Parameters:
    -----------
    estimator : BaseEstimator
        An instance of a scikit-learn classifier to use as the base estimator.

    proba : float, default=0.95
        The minimum probability threshold for pseudo-labeling. Only samples with predicted class probabilities
        greater than or equal to this threshold will be pseudo-labeled.

    max_iter : int, optional
        The maximum number of iterations to run the Pseudo-Labeling algorithm. If not specified, the algorithm will run
        until convergence.

    verbose : int, default=0
        Verbosity level. If 1 or greater, prints information about the fitting process.
    """
    def __init__(self,
                 estimator: BaseEstimator,
                 proba: float = 0.95,
                 max_iter: Optional[int] = None,
                 verbose: int = 0):
        self.estimator = estimator
        self.max_iter = max_iter
        self.verbose = verbose
        self.proba = proba

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series) -> 'PseudoLabeling':
        """
        Fits the Pseudo-Labeling model on the training data.

        Parameters:
        -----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input training data.

        y : pandas Series of shape (n_samples,)
            The target labels for the training data.

        Returns:
        --------
        self : object
            Returns self.
        """

        # Store the data
        self.X_ = X
        self.y_ = y

        # clone the estimator and fit on the data
        self.estimator_ = clone(self.estimator).fit(X, y)

        # initialize the number of iterations
        self.n_iter_ = 1

        return self

    def partial_fit(self,
                    X: pd.DataFrame,
                    y: pd.Series) -> 'PseudoLabeling':
        """
        Incrementally fits the Pseudo-Labeling model on additional data.

        Parameters:
        -----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input data to fit the model on.

        y : pandas Series of shape (n_samples,)
            The target labels for the input data.

        Returns:
        --------
        self : object
            Returns self.
        """

        if hasattr(self, 'estimator_'):
            # concatenate the new data with the existing data
            self.X_ = pd.concat([self.X_, X])
            self.y_ = pd.concat([self.y_, y])

            # fit the estimator on the concatenated data
            self.estimator_.fit(self.X_, self.y_)

            # increment the number of iterations
            self.n_iter_ = self.n_iter_ + 1

        else:
            # fit the model
            self.fit(X, y)

        return self

    def predict(self,
                X: pd.DataFrame) -> np.array:
        """
        Predicts the target labels for the input data.

        Parameters:
        -----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input data to predict labels for.

        Returns:
        --------
        y_pred : numpy array of shape (n_samples,)
            The predicted target labels.
        """
        return self._pseudo_label(X).predict(X)

    def predict_proba(self,
                      X: pd.DataFrame):
        return self._pseudo_label(X).predict_proba(X)

    def decision_function(self,
                          X: pd.DataFrame) -> np.ndarray:
        """
        Predicts class probabilities for the input data.

        Parameters:
        -----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input data to predict class probabilities for.

        Returns:
        --------
        y_proba : numpy array of shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        return self._pseudo_label(X).decision_function(X)

    def _pseudo_label(self,
                      X: pd.DataFrame) -> object:
        """
        Generates pseudo-labels for the input data using the fitted estimator.

        Parameters:
        -----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input data to generate pseudo-labels for.

        Returns:
        --------
        est : object
            The fitted estimator with pseudo-labeled data.
        """

        while not self.max_iter or self.n_iter_ < self.max_iter:

            # Select not added rows
            index = X.index.difference(self.X_.index)
            X_new = X.loc[index]

            # If all rows have been added, break out of the loop
            if not len(index):
                break

            # Predict probabilities
            y_prob = self.estimator_.predict_proba(X_new)
            y_prob = pd.DataFrame(y_prob, index=X_new.index)
            y_new = y_prob.apply(lambda row: row.idxmax(), axis=1)

            # Mask rows with high certainty
            mask = (y_prob >= self.proba).any(axis=1)

            # If no rows have high certainty, break out of the loop
            if not mask.any():
                break

            # Add labeled data & fit
            self.partial_fit(X_new[mask], y_new[mask])

            # Verbose
            if self.verbose:
                logmsg(f"ITER {self.n_iter_}: Add {mask.sum()} labels")

        return self.estimator_

    @if_delegate_has_method(delegate='estimator_')
    def score(self,
              X: pd.DataFrame,
              y: Optional[pd.Series] = None,
              *args,
              **kwargs):
        return self.estimator_.score(X, y, *args, **kwargs)

    # @if_delegate_has_method(delegate='estimator_')
    # def predict(self, X):
    #    return self.estimator_.predict(X)

    # @if_delegate_has_method(delegate='estimator_')
    # def predict_proba(self, X):
    #    return self.estimator_.predict_proba(X)

    # @if_delegate_has_method(delegate='estimator_')
    # def predict_log_proba(self, X):
    #    return self.estimator_.predict_log_proba(X)

    # @if_delegate_has_method(delegate='estimator_')
    # def decision_function(self, X):
    #    return self.estimator_.decision_function(X)

    def __getattr__(self, item):
        return getattr(self.estimator_, item)
