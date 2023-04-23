from collections.abc import Iterable
from typing import Optional, Callable, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted


class MultiTargetRegressor(BaseEstimator, RegressorMixin):
    """Multi target regression

    This strategy consists of fitting one regressor per target. This is a
    simple strategy for extending regressors that do not natively support
    multi-target regression.

    You can use either single estimator, either list of estimators.

    Parameters
    ----------
    estimator : estimator object, or list of estimators, shape (n_targets, )
        An estimator object implementing <fit> and <predict>.
        Or a list of estimators.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for <fit>. None means 1.
        ``-1`` means using all processors.

        When individual estimators are fast to train or predict
        using `n_jobs>1` can result in slower performance due
        to the overhead of spawning processes.

    """
    def __init__(self,
                 estimator: BaseEstimator,
                 scoring: Optional[Union[str, Callable]] = None,
                 weights: Optional[np.array ]= None,
                 n_jobs: Optional[int] = None):
        self.estimator = estimator
        self.scoring = scoring
        self.weights = weights
        self.n_jobs = n_jobs

    def fit(self,
            X: pd.DataFrame,
            Y: pd.DataFrame) -> 'MultiTargetRegressor':
        """
        Fit the model to data.

        Fit a separate model for each output variable.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
        Y : DataFrame, shape (n_samples, n_targets)

        sample_weight : array-like, shape = (n_samples) or None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        Returns
        -------
        self : object

        """

        # Get the column names of the target variables
        self.targets_ = list(Y.columns)

        # Check the estimator and convert to a list if necessary
        self.estimators_ = check_estimator(self.estimator, self.targets_, 'regressor')

        # Fit a separate model for each target variable
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(clone(e), X, Y[target])
            for e, target in zip(self.estimators_, self.targets_))

        return self

    def score(self,
              X: pd.DataFrame,
              Y: pd.DataFrame,
              *args,
              **kwargs) -> float:
        """
        Evaluate the performance of the model on the test data.

        Computes the score for each target variable and returns the
        average score weighted by the `weights` parameter.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
        Y : DataFrame, shape (n_samples, n_targets)
        *args, **kwargs : additional arguments
            Additional arguments to pass to the scorer.

        Returns
        -------
        score : float
            The average score weighted by the `weights` parameter.
        """
        scores = []
        for estimator, target in zip(self.estimators_, self.targets_):
            # Check the scoring metric for the current estimator
            scorer = check_scoring(estimator, self.scoring)

            # Compute the score for the current target variable
            score = scorer(estimator, X, Y[target], *args, **kwargs)

            # Append the score to the list of scores
            scores.append(score)

        # Compute the weighted average of the scores
        return np.average(scores, weights=self.weights)

    @property
    def feature_importances_(self) -> np.array:
        """
        Compute the feature importances of the model.

        Returns
        -------
        imps : array of shape (n_features,)
            Feature importances, averaged across all estimators.
        """

        # Get the feature importances for each estimator
        imps = [e.feature_importances_ for e in self.estimators_]

        # Concatenate the feature importances and average across all estimators
        return np.concatenate(imps).mean(axis=0)

    @property
    def coef_(self) -> np.array:
        """
        Compute the coefficients of the model.

        Returns
        -------
        coefs : array of shape (n_features,)
            Coefficients, averaged across all estimators.
        """

        # Get the coefficients for each estimator
        imps = [e.coef_ for e in self.estimators_]

        # Concatenate the coefficients and average across all estimators
        return np.concatenate(imps).mean(axis=0)

    def predict(self,
                X: pd.DataFrame) -> np.array:
        """
        Predict target values for the given data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Data features.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted target values.
        """
        return _call_estimator(self, X, 'predict')


class MultiTargetClassifier(BaseEstimator, ClassifierMixin):
    """
    Multi target classification

    This strategy consists of fitting one classifier per target. This is a
    simple strategy for extending regressors that do not natively support
    multi-target regression.

    You can use either single estimator, either list of estimators.

    Parameters
    ----------
    estimator : estimator object, or list of estimators, shape (n_targets, )
        An estimator object implementing <fit> and <predict>.
        Or a list of estimators.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for <fit>. None means 1.
        ``-1`` means using all processors.

        When individual estimators are fast to train or predict
        using `n_jobs>1` can result in slower performance due
        to the overhead of spawning processes.

    """

    def __init__(self,
                 estimator: BaseEstimator,
                 scoring: Optional[Union[str, Callable]] = None,
                 weights: Optional[np.array] = None,
                 n_jobs: Optional[int] = None):
        self.estimator = estimator
        self.scoring = scoring
        self.weights = weights
        self.n_jobs = n_jobs

    def fit(self,
            X: pd.DataFrame,
            Y: pd.DataFrame):
        self.targets_ = list(Y.columns)
        self.classes_ = [LabelBinarizer().fit(y).classes_ for _, y in Y.items()]

        self.estimators_ = check_estimator(self.estimator, self.targets_, 'classifier')

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(clone(e), X, Y[target])
            for e, target in zip(self.estimators_, self.targets_))

        return self

    def score(self,
              X: pd.DataFrame,
              Y: pd.DataFrame,
              *args,
              **kwargs) -> float:
        """
        Evaluate the performance of the model on the test data.

        Computes the score for each target variable and returns the
        average score weighted by the `weights` parameter.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            The input data to evaluate the model on.
        Y : DataFrame, shape (n_samples, n_targets)
            The true target values for the input data.
        *args, **kwargs : additional arguments
            Additional arguments to pass to the scorer.

        Returns
        -------
        score : float
            The average score weighted by the `weights` parameter.
        """

        scores = []

        # loop through each estimator and corresponding target variable
        for estimator, target in zip(self.estimators_, self.targets_):
            # get the scorer for the current estimator and scoring metric
            scorer = check_scoring(estimator, self.scoring)

            # compute the score for the current target variable
            score = scorer(estimator, X, Y[target], *args, **kwargs)
            scores.append(score)

        # return the weighted average of the scores
        return np.average(scores, weights=self.weights)

    @property
    def feature_importances_(self) -> np.array:
        """
        Return the feature importances of the model.

        Returns
        -------
        importances : ndarray of shape (n_features,)
            The feature importances of the model.
        """

        # concatenate the feature importances from all estimators and compute their mean
        imps = [e.feature_importances_ for e in self.estimators_]
        return np.concatenate(imps).mean(axis=0)

    @property
    def coef_(self) -> np.array:
        """
        Return the coefficients of the model.

        Returns
        -------
        coef : ndarray of shape (n_features,)
            The coefficients of the model.
        """

        # concatenate the coefficients from all estimators and compute their mean
        imps = [e.coef_ for e in self.estimators_]
        return np.concatenate(imps).mean(axis=0)

    def predict(self,
                X: pd.DataFrame) -> np.array:
        """
        Predict the target values for the input data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            The predicted target values.
        """
        return _call_estimator(self, X, 'predict')

    def predict_proba(self,
                      X: pd.DataFrame) -> np.array:
        """
        Predict the class probabilities for the input data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        proba : ndarray, shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        return _call_estimator(self, X, 'predict_proba')

    def decision_function(self,
                          X: pd.DataFrame) -> np.array:
        """
        Predict the decision function values for the input data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        decision_function : ndarray, shape (n_samples,)
            The decision function values for the input samples.
        """
        return _call_estimator(self, X, 'decision_function')


def _fit_estimator(estimator: BaseEstimator,
                   X: pd.DataFrame,
                   y: pd.Series,
                   sample_weight: np.array = None) -> BaseEstimator:
    """
    Fits an estimator to the input data.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to be fitted.
    X : DataFrame, shape (n_samples, n_features)
        The input data.
    y : Series, shape (n_samples,)
        The target values.
    sample_weight : array-like, shape (n_samples,), optional (default=None)
        Sample weights to be applied during fitting.

    Returns
    -------
    estimator : BaseEstimator
        The fitted estimator.
    """
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator


def _call_estimator(estimator: BaseEstimator,
                    X: pd.DataFrame,
                    method: str) -> list:
    """
    Calls a specified method of the estimator.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator whose method will be called.
    X : DataFrame, shape (n_samples, n_features)
        The input data.
    method : str
        The name of the method to call.

    Returns
    -------
    Y : list
        The output of the method called on each estimator in the ensemble.
    """
    check_is_fitted(estimator, 'estimators_')

    # Define a lambda function to call the specified method on an estimator
    call_estimator = lambda e: getattr(e, method)(X)

    # Call the method on each estimator in the ensemble in parallel
    Y = Parallel(n_jobs=estimator.n_jobs)(delayed(call_estimator)(e)
                                          for e in estimator.estimators_)

    return Y


def check_estimator(estimator: BaseEstimator,
                    targets: list,
                    estimator_type: str = 'regressor') -> list:
    """
    Check if an estimator or a list of estimators match the estimator_type.
    If a list of estimators is passed, check if the number of estimators matches the number of targets.

    Parameters
    ----------
    estimator : BaseEstimator or Iterable
        The estimator(s) to check.
    targets : list
        A list of the target variable(s).
    estimator_type : str, default='regressor'
        The type of estimator to match.

    Returns
    -------
    estimators_list : list
        The list of estimators matching the estimator_type.
    """

    if getattr(estimator, '_estimator_type', None) is estimator_type:
        # If a single estimator is passed and it matches the estimator_type
        estimators_list = [clone(estimator) for _ in targets]

    elif isinstance(estimator, Iterable):
        # If a list of estimators is passed
        estimators_list = estimator

        n_est = len(estimators_list)
        n_tar = len(targets)

        if n_est != n_tar:
            # If the number of estimators does not match the number of targets
            raise ValueError("If passed list of estimators, number of "
                             "estimators \n\t\tshould be equal to Y.shape[1]. "
                             "\n\t\tFound: n_estimators = {}, n_targets = {} "
                             " ".format(n_est, n_tar))

        for i, estimator in enumerate(estimators_list):
            if getattr(estimator, '_estimator_type', None) is not estimator_type:
                # If an estimator in the list does not match the estimator_type
                raise ValueError("If passed list of estimators, each "
                                 "estimator should be {}.\n"
                                 "Error with index {}.".format(estimator_type, i))

    else:
        # If an unknown type of estimator is passed
        raise TypeError("Unknown type of <estimator> passed.\n"
                        "Should be {} or list of {}s."
                        " ".format(estimator_type, estimator_type))

    return estimators_list
