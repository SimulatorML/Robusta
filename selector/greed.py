import pandas as pd
import numpy as np

from sklearn.utils.random import check_random_state
from sklearn.exceptions import NotFittedError

from .base import EmbeddedSelector




class GreedSelector(Selector):
    '''Greed Forward/Backward Feature Selector

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.

    k_features : int or float (default=0.5)
        Number of features to select. If float, interpreted as percentage
        of total # of features:

            - If <forward> is True, <k_features> is maximum # of features.
            - If <forward> is False, <k_features> is minimum # of features.

    forward : boolean (default=True)
        Whether to start from empty set or full set of features:

            - If <forward> is True, add feature on each step
            - If <forward> is False, drop feature on each step

    floating : boolean (default=False)
        Whether to produce step back on each round (if increases score!):

            - If <forward> is True, drop feature on each step
            - If <forward> is False, drop feature on each step

    max_candidates : int, float or NoneType (default=None)
        ...


    '''
    def __init__(self, estimator, k_features=0.5, forward=True, floating=False,
                 max_candidates=None, scoring=None, max_time=None, use_best=True,
                 cv=5, random_state=0, n_jobs=-1, verbose=1, plot=False):

        self.estimator = estimator
        self.k_features = k_features
        self.forward = forward
        self.floating = floating
        self.max_candidates = max_candidates # TODO
        self.max_time = max_time
        self.use_best = use_best

        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.verbose = verbose
        self.plot = plot



    def fit(self, X, y, groups=None):

        self._fit_start(X)
        self._fit(X, y, groups)

        return self



    def partial_fit(self, X, y, groups=None):

        self._fit_start(X, partial=True)
        self._fit(X, y, groups)

        return self



    def _fit_start(self, X, partial=False):

        self.features_ = list(X.columns)
        self.n_features_ = len(self.features_)

        self.k_features_ = _check_k_features(self.k_features, self.n_features_)

        if not partial:
            self.rstate_ = check_random_state(self.random_state)

            if self.forward:
                self.last_subset_ = []
            else:
                self.last_subset_ = self.features_

            self._reset_trials()

        return self



    def _fit(self, X, y, groups):

        all_features = set(self.features_)
        subset = set(self.last_subset_)

        if self.forward:
            is_final = lambda subset: len(subset) >= self.k_features_
            is_start = lambda subset: len(subset) == 1
        else:
            is_final = lambda subset: len(subset) <= self.k_features_
            is_start = lambda subset: len(subset) == self.n_features_

            score = self._eval_subset(subset, X, y, groups)


        while not is_final(subset):

            # Step 1. Step Forward/Backward
            if self.forward:
                subset_updates = all_features - subset
            else:
                subset_updates = subset

            # Evaluate only random subset
            subset_updates = self._subsample_features(subset_updates)

            # Score gain
            feature_scores = pd.Series(None, index=subset_updates)

            for feature in subset_updates:

                # include/exclude (trial)
                if self.forward:
                    candidate = subset | {feature}
                else:
                    candidate = subset - {feature}

                score = self._eval_subset(candidate, X, y, groups)
                feature_scores[feature] = score

            feature_scores = feature_scores.sort_values(ascending=False)
            subset_update = feature_scores.index[0]
            old_score = feature_scores.iloc[0]

            # include/exclude (final)
            if self.forward:
                subset = subset | {subset_update}
            else:
                subset = subset - {subset_update}

            self.last_subset_ = list(subset)

            # stop criteria
            if not self.floating or is_final(subset) or is_start(subset):
                continue

            # Step 2. Step Backward/Forward
            if self.forward:
                subset_updates = subset
            else:
                subset_updates = all_features - subset

            # Evaluate only random subset
            subset_updates = self._subsample_features(subset_updates)

            # Score gain
            feature_scores = pd.Series(None, index=subset_updates)

            for feature in subset_updates:

                # include/exclude (trial)
                if self.forward:
                    candidate = subset - {feature}
                else:
                    candidate = subset | {feature}

                score = self._eval_subset(candidate, X, y, groups)
                feature_scores[feature] = score

            feature_scores = feature_scores.sort_values(ascending=False)
            subset_update = feature_scores.index[0]
            new_score = feature_scores.iloc[0]

            if new_score < old_score:
                continue

            # include/exclude (final)
            if self.forward:
                subset = subset - {subset_update}
            else:
                subset = subset | {subset_update}

            self.last_subset_ = list(subset)

        return self



    def _select_features(self):

        if (self.use_best is True) and hasattr(self, 'best_subset_'):
            return list(self.best_subset_)

        elif (self.use_best is False) and len(self.last_subset_) > 0:
            return list(self.last_subset_)

        else:
            raise NotFittedError('<ExhaustiveSelector> is not fitted')



    def _subsample_features(self, features):

        features = list(features)
        n_features = len(features)

        random_subset = _check_subsample(self.max_candidates, n_features)
        subset = self.rstate_.choice(features, random_subset, replace=False)

        return set(subset)



def _check_subsample(subsample, n_features):
    return n_features
