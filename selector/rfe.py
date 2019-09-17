import pandas as pd
import numpy as np

from sklearn.model_selection import check_cv
from sklearn.exceptions import NotFittedError
from sklearn.base import clone, is_classifier

from robusta.importance import permutation_importance
from robusta.importance import extract_importance

from .base import EmbeddedSelector




class RFE(EmbeddedSelector):
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

    importance_type : {'inbuilt', 'permutation'}, default='permutation'
        Whether to use original estimator's <feature_importances_> or <coef_>
        or use Permutation Importances to measure feature importances.

    min_features : int or float, optional (default=0.5)
        The number of features to select. Float values means percentage of
        features to select. E.g. value 0.5 (by default) means 50% of features.

    step : int or float, optional (default=1)
        The number of features to remove on each step. Float values means
        percentage of left features. E.g. 0.2 means 20% reduction on each step:
        500 => 400 => 320 => ...

    cv : int, cross-validation generator or iterable
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.

    random_state : int or None (default=0)
        Random seed for permutations in PermutationImportance.
        Ignored if <importance_type> set to 'inbuilt'.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int, optional (default=1)
        Verbosity level

    plot : bool, optional (default=False)
        Whether to plot progress

    Attributes
    ----------
    use_cols_ : list of string
        Feature names to select

    n_features_ : int
        Number of selected features

    min_features_ : int
        Minimum number of features

    """

    def __init__(self, estimator, scoring=None, cv=5, min_features=0.5, step=1,
                 n_repeats=5, random_state=0, use_best=True, n_jobs=-1,
                 verbose=1, plot=False):

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
        self.plot


    def fit(self, X, y):

        self._fit_start(X)
        self._fit(X, y)

        return self



    def partial_fit(self, X, y):

        self._fit_start(X, partial=True)
        self._fit(X, y)

        return self



    def _fit_start(self, X, partial=False):

        if not partial:
            self.features_ = list(X.columns)
            self.last_subset_ = self.features_

            self.n_features_ = len(self.features_)
            self.k_features_ = len(self.features_)

            self.min_features_ = _check_min_features(self.min_features, self.n_features_)

            self.rstate_ = check_random_state(self.random_state)

            self._reset_trials()

        return self



    def _fit(self, X, y):

        while True:

            result = self._eval_subset(self.last_subset_, X, y, groups)
            imp = result['importance'].mean(axis=1)

            step = _check_step(self.step, self.min_features, self.k_features_)

            self.last_subset_ = _select_k_best(imp, step, self.min_features_)
            self.k_features_ = len(self.last_subset_)

            if self.k_features_ <= self.min_features:
                result = self._eval_subset(self.last_subset_, X, y, groups)
                imp = result['importance'].mean(axis=1)
                self.feature_importances_ = imp
                break

        return self



    '''def _fit_importance(self, X, y, groups=None):

        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        imps = []

        for trn, oof in cv.split(X, y, groups):
            X_trn, y_trn = X.iloc[trn], y.iloc[trn]
            X_oof, y_oof = X.iloc[oof], y.iloc[oof]

            estimator = clone(self.estimator).fit(X_trn, y_trn)

            imp = self._extract_importance(estimator, X_oof, y_oof)
            imps.append(imp)

        imp = pd.concat(imps, axis=1).mean(axis=1)
        return imp'''



    '''def _extract_importance(self, estimator, X, y):

        if self.importance is 'inbuilt':
            imp = extract_importance(estimator)

        elif self.importance is 'permutation':
            imp = permutation_importance(estimator, X, y,
                                         scoring=self.scoring,
                                         n_repeats=self.n_repeats,
                                         random_state=self.rstate_,
                                         n_jobs=self.n_jobs)'''



    def _select_features(self):

        if hasattr(self, 'best_subset_'):
            if self.use_best:
                return self.best_subset_
            else:
                return self.last_subset_
        else:
            raise NotFittedError('RFE is not fitted')



def _select_k_best(scores, step, min_features):

    n_features = len(scores)
    k_best = max(n_features - step, min_features)

    sort_scores = scores.sort_values(ascending=False)
    best_scores = sort_scores.iloc[:k_best]

    use_cols = list(best_scores.index)
    return use_cols



def _check_min_features(min_features, n_features):

    if isinstance(min_features, int):
        if min_features > 0:
            min_features = min_features
        else:
            raise ValueError('Integer <min_features> must be greater than 0')

    elif isinstance(min_features, float):
        if 0 < min_features < 1:
            min_features = max(1, min_features * n_features)
        else:
            raise ValueError('Float <min_features> must be from interval (0, 1)')

    else:
        raise ValueError('Parameter <min_features> must be int or float, \
                         got {}'.format(min_features))

    return min_features



def _check_step(step, n_features):

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
        raise ValueError('Parameter <step> must be int or float, \
                         got {}'.format(step))

    return step
