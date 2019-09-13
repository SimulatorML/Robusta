import pandas as pd
import numpy as np

from sklearn.utils.random import check_random_state
from sklearn.exceptions import NotFittedError
from sklearn.base import clone, is_classifier

from robusta.crossval import crossval_score

from .base import EmbeddedSelector




class RandomSubset(EmbeddedSelector):
    '''Random Subset Feature Selector

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.

    min_features, max_features : int or float
        Minimum & maximum number of features. If float, interpreted as
        percentage of total number of features. <max_features> must be greater
        or equal to the <min_features>.

    max_trials : int or None
        Maximum number of iterations. None for no limits. Use <max_time>
        or Ctrl+C for KeyboardInterrupt to stop optimization in this case.

    max_time : float or None
        Maximum time (in seconds). None for no limits. Use <max_trials>
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

    random_state : int
        Random state for cross-validation instance

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel. None means 1.

    '''

    def __init__(self, estimator, min_features=0.5, max_features=0.9, scoring=None,
                 max_trials=20, max_time=None, cv=5, random_state=0, n_jobs=None):

        self.estimator = estimator
        self.min_features = min_features
        self.max_features = max_features
        self.max_trials = max_trials
        self.max_time = max_time

        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs



    def fit(self, X, y, groups=None):

        X_cols = list(X.columns)
        n_features = len(X_cols)

        self.min_features_ = _check_k_features(self.min_features, n_features)
        self.max_features_ = _check_k_features(self.max_features, n_features)

        if self.min_features_ < self.max_features_:
            raise ValueError('<max_features> must not be less than <min_features>')


        rstate = check_random_state(self.random_state)
        weights = nCk_range(self.min_features_, self.max_features_, n_features, rstate)

        while True:
            try:
                k_cols = weighted_choice(weights)
                subset = rstate.choice(X_cols, k_cols, replace=False)

                score = self._eval_subset(subset, X, y, groups)

            except KeyboardInterrupt:
                break

        return self



    def _select_features(self):

        if hasattr(self, 'best_subset_'):
            return self.best_subset_
        else:
            raise NotFittedError('RSFS is not fitted')




def _check_k_features(k_features, n_features):

    if isinstance(k_features, int):
        if k_features < 1:
            raise ValueError('Parameters <min_features> & <max_features> must be \
                              integer (greater than 0) or float (0..1)')

    elif isinstance(k_features, float):
        if 0 < k_features < 1:
            k_features = max(1, k_features * n_features)
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


def nCk_range(k_min, k_max, n):
    k_range = range(k_min, k_max+1)

    C = [nCk(n, k) for k in k_range]
    return pd.Series(C, index=k_range).sort_index()


def weighted_choice(weights):
    # weights ~ pd.Series
    rnd = np.random.random() * sum(weights)
    for i, w in weights.items():
        rnd -= w
        if rnd <= 0:
            return i
