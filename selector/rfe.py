import pandas as pd
import numpy as np

from sklearn.model_selection import check_cv
from sklearn.exceptions import NotFittedError
from sklearn.base import clone, is_classifier

from robusta.importance import extract_importance
#from robusta.crossval import crossval

from .base import Selector

# Original: sklearn.feature_selection.SelectFromModel



class RFE(Selector):
    """Meta-transformer for selecting features based on importance weights.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        This can be both a fitted (if cv='prefit') or a non-fitted estimator.
        The estimator must have either a <feature_importances_> or <coef_>
        attribute after fitting.

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

            - None, to disable cross-validation and train single estimator
            on whole dataset (default).
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.

    Attributes
    ----------
    use_cols_ : list of string
        Feature names to select

    n_features_ : int
        Number of selected features

    min_features_ : int
        Minimum number of features

    """

    def __init__(self, estimator, min_features=0.5, step=1, cv=None):

        self.estimator = estimator
        self.min_features = min_features
        self.step = step
        self.cv = cv



    def fit(self, X, y, groups=None):

        self.use_cols_ = list(X.columns)
        self.n_features_ = len(self.use_cols_)

        self.min_features_ = _check_min_features(self.min_features, self.n_features_)

        while True:

            if self.cv is None:
                imp = self._fit_importance(X[self.use_cols_], y)
            else:
                imp = self._cv_fit_importance(X[self.use_cols_], y, groups)

            step = _check_step(self.step, self.n_features_)

            self.use_cols_ = _select_k_best(imp, step, self.min_features_)
            self.n_features_ = len(self.use_cols_)

            if self.n_features_ <= self.min_features_:
                break

        return self



    def _fit_importance(self, X, y):

        estimator = clone(self.estimator).fit(X, y)
        imp = extract_importance(estimator)

        return imp



    def _cv_fit_importance(self, X, y, groups=None):

        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        imps = []

        for trn, _ in cv.split(X, y, groups):
            X_trn, y_trn = X.iloc[trn], y.iloc[trn]

            estimator = clone(self.estimator).fit(X_trn, y_trn)
            imp = extract_importance(estimator)
            imps.append(imp)

        imp = pd.concat(imps, axis=1).mean(axis=1)
        return imp



    def _select_features(self):

        if hasattr(self, 'use_cols_'):
            return self.use_cols_
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
            step = max(1, step * n_features)
            step = int(step)
        else:
            raise ValueError('Float <step> must be from interval (0, 1)')

    else:
        raise ValueError('Parameter <step> must be int or float, \
                         got {}'.format(step))

    return step
