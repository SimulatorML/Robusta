from typing import Optional, Callable, Union

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ..utils import all_subsets
from . import _WrappedSelector, _WrappedGroupSelector


class ExhaustiveSelector(_WrappedSelector):
    """
    Exhaustive feature selector for sampling and evaluating all possible
    feature subsets of specified size.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.

    cv : int, cross-validation generator or iterable
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.

    scoring : string, callable or None, optional, default: None
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only a single value.
        If None, the estimator's default scorer (if available) is used.

    min_features, max_features : int or float
        Minimum & maximum number of features. If float, interpreted as
        percentage of total number of features. <max_features> must be greater
        or equal to the <min_features>.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int, optional (default=1)
        Verbosity level

    Attributes
    ----------
    features_ : list of string
        Feature names

    n_features_ : int
        Total number of features

    min_features_, max_features_ : int
        Minimum and maximum subsets size

    weights_ : Series
        Subset sizes weights (not normalized)

    rstate_ : object
        Random state instance

    trials_ : DataFrame
        All evaluated subsets:

            - 'subset': subset of feature names
            - 'score': average cross-validation score
            - 'time': fitting time

    best_iter_: int
        Best trial's index

    best_score_: float
        Best trial's score

    best_subset_: list of string
        Best subset of features

    total_time_: float
        Total optimization time (seconds)

    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cv: int = 5,
        scoring: Optional[Union[str, Callable]] = None,
        min_features: float = 0.5,
        n_jobs: int = -1,
        max_features: float = 0.9,
        verbose: int = 1,
        n_digits: int = 4,
        cv_kwargs: Optional[dict] = None,
    ):
        if cv_kwargs is None:
            cv_kwargs = {}
        self.estimator = estimator
        self.min_features = min_features
        self.max_features = max_features

        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

        self.verbose = verbose
        self.n_digits = n_digits

        self.cv_kwargs = cv_kwargs

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None
    ) -> "ExhaustiveSelector":
        """
        Fits the selector on the input data and returns the fitted object.

        Parameters
        ----------
        X : DataFrame
            The input data frame containing the features.
        y : Series
            The target variable.
        groups : Series
            The groups of the features.

        Returns
        -------
        self : object
            ExhaustiveSelector: The fitted selector object.
        """

        # initialize the selector and perform the initial fitting steps
        self._fit_start(X)
        self._fit(X, y, groups)

        return self

    def partial_fit(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None
    ) -> "ExhaustiveSelector":
        """
        Fits the selector on a subset of the input data and returns the partially fitted object.

        Parameters
        ----------
        X : DataFrame
            The input data frame containing the features.
        y : Series
            The target variable.
        groups : Series
            The groups of the features.

        Returns
        -------
        self : object
            ExhaustiveSelector: The partially fitted selector object.
        """

        # initialize the selector and perform the initial fitting steps for partial fitting
        self._fit_start(X, partial=True)
        self._fit(X, y, groups)

        return self

    def _fit_start(
        self, X: pd.DataFrame, partial: bool = False
    ) -> "ExhaustiveSelector":
        """
        Initializes the selector object and sets the features of the input data frame.

        Parameters
        ----------
        X : DataFrame
            The input data frame containing the features.
        partial : bool
            A flag indicating whether the fitting is partial or not.

        Returns
        -------
        self : object
            ExhaustiveSelector: The selector object with the features set.
        """

        # set the features of the input data frame
        self._set_features(X)

        # if the fitting is not partial, generate all possible feature subsets
        if not partial:
            k_range = range(self.min_features_, self.max_features_ + 1)
            self.subsets_ = all_subsets(self.features_, k_range)
            self.subsets_ = list(self.subsets_)
            self.max_iter = len(self.subsets_)
            self._reset_trials()

        # if the fitting is partial, reset the iteration counter
        if not hasattr(self, "k_iter") or not partial:
            self.k_iter = 0

        return self

    def _fit(
        self, X: pd.DataFrame, y: pd.Series, groups: pd.Series
    ) -> "ExhaustiveSelector":
        """
        Fits the selector object using the wrapped group selector algorithm.

        Parameters
        ----------
        X : DataFrame
            The input data frame containing the features.
        y : Series
            The target variable.
        groups : Series
            The groups of the features.

        Returns
        -------
        self : object
            ExhaustiveSelector: The fitted selector object.
        """

        # iterate over all possible feature subsets and evaluate their performance
        while self.k_iter < self.max_iter:
            subset = self.subsets_[self.k_iter]
            try:
                self.eval_subset(subset, X, y, groups)
            except KeyboardInterrupt:
                break
            self.k_iter += 1
        return self

    def get_subset(self) -> None:
        """
        Returns the best subset of features found by the model.

        Returns
        -------
        None:
            None or The best subset found by the model.
        """

        if hasattr(self, "best_subset_"):
            return self.best_subset_
        else:
            model_name = self.__class__.__name__
            raise NotFittedError(f"{model_name} is not fitted")


class GroupExhaustiveSelector(_WrappedGroupSelector, ExhaustiveSelector):
    pass
