import pandas as pd
import numpy as np

from sklearn.utils.random import check_random_state
from sklearn.exceptions import NotFittedError

from robusta.utils import logmsg

from .base import _AgnosticSelector, _GroupSelector, _check_k_features




class GreedSelector(_AgnosticSelector):
    '''Greed Forward/Backward Feature Selector

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

    forward : boolean (default=True)
        Whether to start from empty set or full set of features:

            - If <forward> is True, add feature on each step
            - If <forward> is False, drop feature on each step

    floating : boolean (default=False)
        Whether to produce step back on each round (if increases score!):

            - If <forward> is True, drop feature on each step
            - If <forward> is False, drop feature on each step

    k_features : int or float (default=0.5)
        Number of features to select. If float, interpreted as percentage
        of total # of features:

            - If <forward> is True, <k_features> is maximum # of features.
            - If <forward> is False, <k_features> is minimum # of features.

    max_iter : int or None
        Maximum number of iterations. None for no limits. Use <max_time>
        or Ctrl+C for KeyboardInterrupt to stop optimization in this case.

    max_time : float or None
        Maximum time (in seconds). None for no limits. Use <max_iter>
        or Ctrl+C for KeyboardInterrupt to stop optimization in this case.

    use_best : bool (default=True)
        Whether to use subset with best score or last selected subset.

    random_state : int or None (default=0)
        Random seed for permutations in PermutationImportance.
        Ignored if <importance_type> set to 'inbuilt'.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int, optional (default=1)
        Verbosity level

    n_digits : int (default=4)
        Verbose score(s) precision

    '''
    def __init__(self, estimator, cv=5, scoring=None, forward=True, floating=False,
                 k_features=0.5, max_time=None, use_best=True, random_state=0,
                 n_jobs=None, verbose=1, n_digits=4):

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
        self.k_features_ = _check_k_features(self.k_features, self.n_features_, 'k_features')

        if not partial:

            self.rstate_ = check_random_state(self.random_state)
            self.subset_ = self.features_.copy()

            if self.forward:
                self.subset_.set_subset([])

            self._reset_trials()

        return self



    def _fit(self, X, y, groups):

        if self.forward:
            is_final = lambda subset: len(subset) >= self.k_features_
        else:
            is_final = lambda subset: len(subset) <= self.k_features_

            self._eval_subset(self.subset_, X, y, groups)
            self.score_ = self.subset_.score


        while not is_final(self.subset_):

            # STEP 1. Step Forward/Backward
            if self.verbose:
                logmsg('STEP {}'.format('FORWARD' if self.forward else 'BACKWARD'))

            if self.forward:
                updates = self.features_.remove(*self.subset_)
            else:
                updates = self.subset_

            # Find Next Best Update
            score  = -np.inf
            subset = None

            for feature in updates:

                # Include/Exclude Feature
                if self.forward:
                    candidate = self.subset_.append(feature)
                else:
                    candidate = self.subset_.remove(feature)

                candidate.parents = (self.subset_, )

                # Evaluate Candidate
                try:
                    self._eval_subset(candidate, X, y, groups)

                    if candidate.score > score:
                        score  = candidate.score
                        subset = candidate

                except KeyboardInterrupt:
                    raise

                except:
                    pass

            # Update Subset
            self.subset_ = subset
            self.score_  = score

            # Stop Criteria
            if not self.floating or is_final(subset):
                continue


            # STEP 2. Step Backward/Forward
            if self.verbose:
                logmsg('STEP {}'.format('BACKWARD' if self.forward else 'FORWARD'))

            if not self.forward:
                updates = self.features_.remove(*self.subset_)
            else:
                updates = self.subset_

            # Find Next Best Update
            score  = -np.inf
            subset = None

            for feature in updates:

                # Exclude/Include Feature
                if not self.forward:
                    candidate = self.subset_.append(feature)
                else:
                    candidate = self.subset_.remove(feature)

                candidate.parents = (self.subset_, )

                # Check if Already Exsists
                if candidate in self.trials_:
                    continue

                # Evaluate Candidate
                try:
                    self._eval_subset(candidate, X, y, groups)

                    if candidate.score > score:
                        score  = candidate.score
                        subset = candidate

                except KeyboardInterrupt:
                    raise

                except:
                    pass

            # Stop Criteria
            if score < self.score_:
                continue

            # Update Subset
            self.subset_ = subset
            self.score_  = score

        return self


    def get_subset(self):

        if (self.use_best is True) and hasattr(self, 'best_subset_'):
            return self.best_subset_

        elif (self.use_best is False) and len(self.subset_) > 0:
            return self.last_subset_

        else:
            model_name = self.__class__.__name__
            raise NotFittedError('{} is not fitted'.format(model_name))



class GroupGreedSelector(_GroupSelector, GreedSelector):
    pass
