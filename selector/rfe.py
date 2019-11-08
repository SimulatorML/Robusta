import pandas as pd
import numpy as np

from time import time

from sklearn.utils.random import check_random_state
from sklearn.model_selection import check_cv
from sklearn.exceptions import NotFittedError
from sklearn.base import clone, is_classifier

from robusta.importance import *

from .base import _AgnosticSelector, _GroupSelector




class RFE(_AgnosticSelector):
    """Feature ranking with recursive feature elimination (RFE) and
    cross-validated selection of the best number of features.

    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of recursive feature elimination
    (RFE) is to select features by recursively considering smaller and smaller
    sets of features. First, the estimator is trained on the initial set of
    features and the importance of each feature is obtained either through a
    <coef_> attribute or through a <feature_importances_> attribute. Then, the
    least important features are pruned from current set of features. That
    procedure is recursively repeated on the pruned set until the desired
    number of features to select is eventually reached.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        The estimator must have either a <feature_importances_> or <coef_>
        attribute after fitting.

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

    min_features : int or float, optional (default=0.5)
        The number of features to select. Float values means percentage of
        features to select. E.g. value 0.5 (by default) means 50% of features.

    step : int or float, optional (default=1)
        The number of features to remove on each step. Float values means
        percentage of left features. E.g. 0.2 means 20% reduction on each step:
        500 => 400 => 320 => ...

    random_state : int or None (default=0)
        Random seed for permutations in PermutationImportance.
        Ignored if <importance_type> set to 'inbuilt'.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int, optional (default=1)
        Verbosity level

    Attributes
    ----------
    use_cols_ : list of string
        Feature names to select

    n_features_ : int
        Number of selected features

    min_features_ : int
        Minimum number of features

    """

    def __init__(self, estimator, cv=5, scoring=None, min_features=0.5, step=1,
                 use_best=True, n_jobs=None, verbose=1, n_digits=4):

        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv

        self.min_features = min_features
        self.step = step
        self.use_best = use_best

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

        if not partial:

            self._set_features(X)
            self.subset_ = self.features_

            self._reset_trials()

        self.k_range_ = []
        k_features = self.n_features_

        while k_features > self.min_features_:
            step = _check_step(self.step, k_features, self.min_features_)
            k_features = k_features - step
            self.k_range_.append(k_features)

        self.max_iter = len(self.k_range_) + getattr(self, 'n_iters_', 0) + 1
        self.k_range_ = iter(self.k_range_)

        return self


    @property
    def k_features_(self):
        return len(self.subset_)


    @property
    def forward(self):
        return False


    def _fit(self, X, y, groups):

        self._eval_subset(self.subset_, X, y, groups)

        for k in self.k_range_:
            try:
                scores = self.subset_.importance
                subset = _select_k_best(scores, k)
                parent = self.subset_

                self.subset_ = self.subset_.copy().set_subset(subset)
                self.subset_.parents = [parent]

                self._eval_subset(self.subset_, X, y, groups)

                if self.k_features_ <= self.min_features_:
                    break

            except KeyboardInterrupt:
                break

        return self


    def get_subset(self):

        if (self.use_best is True) and self.n_iters_ > 0:
            return self.best_subset_

        elif (self.use_best is False) and len(self.subset_) > 0:
            return self.subset_

        else:
            model_name = self.__class__.__name__
            raise NotFittedError(f'{model_name} is not fitted')






class PermutationRFE(RFE):

    def __init__(self, estimator, cv=5, scoring=None, min_features=0.5, step=1,
                 n_repeats=5, random_state=0, use_best=True, n_jobs=None,
                 verbose=1, n_digits=4):

        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv

        self.min_features = min_features
        self.step = step

        self.n_repeats = n_repeats
        self.random_state = random_state
        self.use_best = use_best

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_digits = n_digits


    def _eval_subset(self, subset, X, y, groups=None):

        progress_bar = (self.verbose >= 5)
        subset = list(subset)

        perm = PermutationImportance(self.estimator, self.cv, self.scoring,
                                     self.n_repeats, n_jobs=self.n_jobs,
                                     random_state=self.random_state,
                                     progress_bar=progress_bar)
        perm.fit(X[subset], y, groups)

        subset.score     = np.mean(perm.scores_)
        subset.score_std =  np.std(perm.scores_)

        subset.importance     = perm.feature_importances_
        subset.importance_std = perm.feature_importances_std_

        return subset



class GroupPermutationRFE(_GroupSelector, PermutationRFE):

    def _eval_subset(self, subset, X, y, groups=None):

        progress_bar = (self.verbose >= 5)

        perm = GroupPermutationImportance(self.estimator, self.cv, self.scoring,
                                          self.n_repeats, n_jobs=self.n_jobs,
                                          random_state=self.random_state,
                                          progress_bar=progress_bar)
        perm.fit(X[subset], y, groups)

        subset.score     = np.mean(perm.scores_)
        subset.score_std =  np.std(perm.scores_)

        subset.importance     = perm.feature_importances_
        subset.importance_std = perm.feature_importances_std_

        return subset




def _select_k_best(subset, k_best):
    return scores.index[np.argsort(-scores.values)][:k_best]




def _check_step(step, n_features, k_features):

    if isinstance(step, int):
        if step > 0:
            step = step
        else:
            raise ValueError('Integer <step> must be greater than 0')

    elif isinstance(step, float):
        if 0 < step < 1:
            step = max(step * n_features, 1)
            step = int(step)
        else:
            raise ValueError('Float <step> must be from interval (0, 1)')

    else:
        raise ValueError(f'Parameter <step> must be int or float, got {step}')

    return min(step, int(n_features-k_features))
