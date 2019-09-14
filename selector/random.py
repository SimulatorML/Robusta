import pandas as pd
import numpy as np

from sklearn.utils.random import check_random_state
from sklearn.exceptions import NotFittedError

from .base import EmbeddedSelector




class RandomSubset(EmbeddedSelector):
    '''Random feature selector for sampling and evaluating randomly choosen
    feature subsets of specified size

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.

    min_features, max_features : int or float
        Minimum & maximum number of features. If float, interpreted as
        percentage of total number of features. <max_features> must be greater
        or equal to the <min_features>.

    max_iter : int or None
        Maximum number of iterations. None for no limits. Use <max_time>
        or Ctrl+C for KeyboardInterrupt to stop optimization in this case.

    max_time : float or None
        Maximum time (in seconds). None for no limits. Use <max_iter>
        or Ctrl+C for KeyboardInterrupt to stop optimization in this case.

    scoring : string, callable or None, optional, default: None
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only a single value.
        If None, the estimator's default scorer (if available) is used.

    cv : int, cross-validation generator or iterable
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.

    weights : {'binomal', 'uniform'}
        Probability for subset sizes:

            - 'uniform': each # of features selected with equal probability
            - 'binomal': each # of features selected with probability, which
            proportional to # of different subsets of given size (binomal
            coefficient nCk, where n - total # of features, k - subset size)

    random_state : int
        Random state for subsets generator

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int, optional (default=1)
        Verbosity level

    plot : bool, optional (default=False)
        Whether to plot progress

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

    def __init__(self, estimator, min_features=0.5, max_features=0.9, scoring=None,
                 max_iter=20, max_time=None, cv=5, weights='uniform',
                 random_state=0, n_jobs=-1, verbose=1, plot=False):

        self.estimator = estimator
        self.min_features = min_features
        self.max_features = max_features
        self.max_iter = max_iter
        self.max_time = max_time
        self.weights = weights

        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.verbose = verbose
        self.plot = plot



    def fit(self, X, y, groups=None):

        self._fit(X)

        while True:
            try:
                k = weighted_choice(self.weights_, self.rstate_)
                subset = self.rstate_.choice(self.features_, k, replace=False)

                score = self._eval_subset(subset, X, y, groups)

            except KeyboardInterrupt:
                break

        return self



    def partial_fit(self, X, y, groups=None):

        self._fit(X, partial=True)

        while True:
            try:
                k = weighted_choice(self.weights_, self.rstate_)
                subset = self.rstate_.choice(self.features_, k, replace=False)

                self._eval_subset(subset, X, y, groups)

            except KeyboardInterrupt:
                break

        return self



    def _fit(self, X, partial=False):

        if not partial:
            self._reset_trials()

        if not partial and hasattr(self, 'random_state'):
            self.rstate_ = check_random_state(self.random_state)

        self.features_ = list(X.columns)
        self.n_features_ = len(self.features_)

        self.min_features_ = _check_k_features(self.min_features, self.n_features_)
        self.max_features_ = _check_k_features(self.max_features, self.n_features_)

        if self.min_features_ > self.max_features_:
            raise ValueError('<max_features> must not be less than <min_features>')

        weights_values = ['uniform', 'binomal']

        if self.weights is 'binomal':
            self.weights_ = binomal_weights(self.min_features_,
                                            self.max_features_,
                                            self.n_features_)
        elif self.weights is 'uniform':
            self.weights_ = uniform_weights(self.min_features_,
                                            self.max_features_)
        else:
            raise ValueError('<weights> must be from {}'.format(weights_values))

        return self



    def _select_features(self):

        if hasattr(self, 'best_subset_'):
            return self.best_subset_
        else:
            raise NotFittedError('<RandomSubset> is not fitted')





def _check_k_features(k_features, n_features):

    if isinstance(k_features, int):
        if k_features < 1:
            raise ValueError('Parameters <min_features> & <max_features> must be \
                              integer (greater than 0) or float (0..1)')

    elif isinstance(k_features, float):
        if 0 < k_features < 1:
            k_features = max(k_features * n_features, 1)
            k_features = int(k_features)
        else:
            raise ValueError('Parameters <min_features> & <max_features> must be \
                              integer (greater than 0) or float (0..1)')

    else:
        raise ValueError('Parameters <min_features> & <max_features> must be \
                          integer (greater than 0) or float (0..1)')

    return k_features




fact = lambda x: x*fact(x-1) if x else 1


def nCk(n, k):
    return fact(n) // fact(k) // fact(n-k)


def binomal_weights(k_min, k_max, n):
    k_range = range(k_min, k_max+1)
    C = [nCk(n, k) for k in k_range]
    return pd.Series(C, index=k_range).sort_index()


def uniform_weights(k_min, k_max):
    k_range = range(k_min, k_max+1)
    return pd.Series(1, index=k_range).sort_index()


def weighted_choice(weights, rstate):
    # weights ~ pd.Series
    rnd = rstate.uniform() * weights.sum()
    for i, w in weights.items():
        rnd -= w
        if rnd <= 0:
            return i
