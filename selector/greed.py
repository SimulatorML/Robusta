import pandas as pd
import numpy as np

from sklearn.utils.random import check_random_state
from sklearn.exceptions import NotFittedError

from robusta.utils import logmsg

from .base import SequentialAgnosticSelector




class GreedSelector(SequentialAgnosticSelector):
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
    def __init__(self, estimator, cv=5, scoring=None, forward=True, floating=False,
                 k_features=0.5, max_candidates=None, max_time=None, use_best=True,
                 random_state=0, n_jobs=None, verbose=1, n_digits=4, plot=False):

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
        self.n_digits = n_digits
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

            trial = self._eval_subset(subset, X, y, groups)
            score = trial['score']


        while not is_final(subset):

            # Step 1. Step Forward/Backward
            if self.verbose:
                logmsg('STEP {}'.format('FORWARD' if self.forward else 'BACKWARD'))

            if self.forward:
                subset_updates = all_features - subset
            else:
                subset_updates = subset

            # Evaluate only random subset
            subset_updates = self._subsample_features(subset_updates)

            # Score gain
            feature_scores = pd.Series(np.nan, index=subset_updates)

            for feature in subset_updates:

                # include/exclude (trial)
                if self.forward:
                    candidate = subset | {feature}
                else:
                    candidate = subset - {feature}

                try:
                    result = self._eval_subset(candidate, X, y, groups, prev_subset=subset)
                    feature_scores[feature] = np.mean(result['score'])
                except KeyboardInterrupt:
                    raise
                except:
                    feature_scores[feature] = np.nan

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
            if self.verbose:
                logmsg('STEP {}'.format('BACKWARD' if self.forward else 'FORWARD'))

            if self.forward:
                subset_updates = subset
            else:
                subset_updates = all_features - subset

            # Evaluate only random subset
            subset_updates = self._subsample_features(subset_updates)

            # Score gain
            feature_scores = pd.Series(np.nan, index=subset_updates)

            for feature in subset_updates:

                # include/exclude (trial)
                if self.forward:
                    candidate = subset - {feature}
                else:
                    candidate = subset | {feature}

                if self._find_trial(candidate):
                    continue

                try:
                    result = self._eval_subset(candidate, X, y, groups, prev_subset=subset)
                    feature_scores[feature] = np.mean(result['score'])
                except KeyboardInterrupt:
                    raise
                except:
                    feature_scores[feature] = np.nan

            if not (feature_scores > old_score).any():
                continue

            feature_scores = feature_scores.dropna()
            feature_scores = feature_scores.sort_values(ascending=False)

            subset_update = feature_scores.index[0]

            # include/exclude (final)
            if self.forward:
                subset = subset - {subset_update}
            else:
                subset = subset | {subset_update}

            self.last_subset_ = list(subset)

        return self


    def get_features(self):

        if (self.use_best is True) and hasattr(self, 'best_subset_'):
            return list(self.best_subset_)

        elif (self.use_best is False) and len(self.last_subset_) > 0:
            return list(self.last_subset_)

        else:
            model_name = self.__class__.__name__
            raise NotFittedError('{} is not fitted'.format(model_name))



    def _subsample_features(self, features):

        # TODO: random subsample

        #features = list(features)
        #n_features = len(features)

        #random_subset = _check_subsample(self.max_candidates, n_features)
        #subset = self.rstate_.choice(features, random_subset, replace=False)

        #return set(subset)
        return features



def _check_k_features(k_features, n_features):

    if isinstance(k_features, int):
        if k_features < 1:
            raise ValueError('Parameters <k_features> must be integer \
                              (greater than 0) or float (0..1)')

    elif isinstance(k_features, float):
        if 0 < k_features < 1:
            k_features = max(k_features * n_features, 1)
            k_features = int(k_features)
        else:
            raise ValueError('Parameters <k_features> must be integer \
                              (greater than 0) or float (0..1)')

    else:
        raise ValueError('Parameters <k_features> must be integer \
                          (greater than 0) or float (0..1)')

    return k_features



def _check_subsample(subsample, n_features):
    return n_features
