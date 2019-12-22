import pandas as pd
import numpy as np

from sklearn.exceptions import NotFittedError

from robusta.utils import all_subsets
from .base import _WrappedSelector, _GroupSelector



class ExhaustiveSelector(_WrappedSelector):
    '''Exhaustive feature selector for sampling and evaluating all possible
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

    '''

    def __init__(self, estimator, cv=5, scoring=None, min_features=0.5, n_jobs=-1,
                 max_features=0.9, verbose=1, n_digits=4):

        self.estimator = estimator
        self.min_features = min_features
        self.max_features = max_features

        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

        self.verbose = verbose
        self.n_digits = n_digits



    def fit(self, X, y, groups=None):

        self._fit_start(X)
        self._fit(X, y, groups)

        return self



    def partial_fit(self, X, y, groups=None):

        self._fit_start(X, partial=True)
        self._fit(X, y, groups)

        return self



    def _fit_start(self, X, partial=False):

        self._set_features(X)

        if not partial:
            k_range = range(self.min_features_, self.max_features_+1)
            self.subsets_ = all_subsets(self.features_, k_range)
            self.subsets_ = list(self.subsets_)
            self.max_iter = len(self.subsets_)
            self._reset_trials()

        if not hasattr(self, 'k_iter') or not partial:
            self.k_iter = 0

        return self



    def _fit(self, X, y, groups):

        while self.k_iter < self.max_iter:

            subset = self.subsets_[self.k_iter]

            try:
                self.eval_subset(subset, X, y, groups)
            except KeyboardInterrupt:
                break

            self.k_iter += 1

        return self

    def get_subset(self):

        if hasattr(self, 'best_subset_'):
            return self.best_subset_
        else:
            model_name = self.__class__.__name__
            raise NotFittedError(f'{model_name} is not fitted')



class GroupExhaustiveSelector(_GroupSelector, ExhaustiveSelector):
    pass
