import numpy as np
import pandas as pd

from robusta.selector.base import _WrappedSelector, _WrappedGroupSelector, _check_k_features
from sklearn.utils.random import check_random_state


def perturb_subset(subset, step, random_state=None, drop_attrs=['score']):

    rstate = check_random_state(random_state)
    update = rstate.choice(subset.features, step, False)

    del_list = set(subset) & set(update)
    add_list = set(update) - set(subset)

    subset_ = subset.copy()
    subset_ = subset_.remove(*del_list)
    subset_ = subset_.append(*add_list)

    for attr in drop_attrs:
        if hasattr(subset_, attr):
            delattr(subset_, attr)

    subset_.parents = (subset, )
    return subset_


class SAS(_WrappedSelector):

    def __init__(self, estimator, cv=5, scoring=None, min_step=0.01, max_step=0.05,
                 min_features=0.1, max_features=0.9, max_iter=50, temp=1.0,
                 random_state=None, n_jobs=None, verbose=1, n_digits=4):

        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring

        self.min_features = min_features
        self.max_features = max_features
        self.min_step = min_step
        self.max_step = max_step
        self.max_iter = max_iter
        self.temp = temp

        self.random_state = random_state
        self.verbose = verbose
        self.n_digits = n_digits
        self.n_jobs = n_jobs


    def fit(self, X, y, groups=None):
        self._fit_start(X, y, groups)
        self._fit(X, y, groups)
        return self


    def partial_fit(self, X, y, groups=None):
        self._fit(X, y, groups)
        return self


    def _fit_start(self, X, y, groups):

        # Basic
        self.rstate_ = check_random_state(self.random_state)
        self._set_features(X)
        self._reset_trials()

        # First trial
        k_min = self.min_features_
        k_max = self.max_features_
        k = self.rstate_.choice(range(k_min, k_max+1))
        subset = self.features_.sample(size=k, random_state=self.rstate_)

        self.eval_subset(subset, X, y, groups)
        self.subset_ = subset

        return self


    def _fit(self, X, y, groups=None):

        while self.n_iters_ < self.max_iter:
            try:
                # Pertrub the current subset
                k_min = self.min_step_
                k_max = self.max_step_
                k = self.rstate_.choice(range(k_min, k_max+1))
                subset = perturb_subset(self.subset_, k, self.rstate_)

                # Evaluate perfomance
                self.eval_subset(subset, X, y, groups)
                old_score = self.subset_.score
                new_score = subset.score

                if new_score > old_score:
                    self.subset_ = subset

                else:
                    # Acceptance probability
                    temp = self.temp * self.max_iter / self.n_iters_
                    diff = (old_score - new_score) / abs(old_score)
                    prob = np.exp(-diff/temp)

                    if self.rstate_.rand() < prob:
                        self.subset_ = subset

            except KeyboardInterrupt:
                break

        return self


    @property
    def min_step_(self):
        min_step = _check_k_features(self.min_step,
                                     self.n_features_,
                                     'min_step')
        return min_step


    @property
    def max_step_(self):
        max_step = _check_k_features(self.max_step,
                                     self.n_features_,
                                     'max_step')
        return max_step


class GroupSAS(_WrappedGroupSelector, SAS):
    pass
