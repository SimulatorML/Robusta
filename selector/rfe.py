import pandas as pd
import numpy as np

from sklearn.model_selection import check_cv
from sklearn.exceptions import NotFittedError
from sklearn.base import clone, is_classifier

from robusta.importance import extract_importance
from robusta.crossval import crossval

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

    threshold : string, float, optional (default None)
        The threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded. If "median" (resp. "mean"), then the <threshold> value is
        the median (resp. the mean) of the feature importances. A scaling
        factor (e.g., "1.25*mean") may also be used. If None, drop features
        only based on <max_features>.

    max_features : int, float or None, optional (default 0.5)
        The maximum number of features selected scoring above <threshold>.
        If float, interpreted as proportion of all features.

        To disable <threshold> and only select based on <max_features>,
        set <threshold> to -np.inf.

    cv : int, cross-validation generator, iterable or "prefit"
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

            - None, to disable cross-validation and train single estimator
            on whole dataset (default).
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.
            - "prefit" string constant.

        If "prefit" is passed, it is assumed that <estimator> has been
        fitted already and <fit> function will raise error.

    Attributes
    ----------
    estimator_ : list of fitted estimators, or single fitted estimator
        If <cv> is 'prefit'. If <cv> is None, return single estimator.
        Otherwise return list of fitted estimators, length (n_folds, ).

    feature_importances_ : Series of shape (n_features, )
        Feature importances, extracted from estimator(s)

    threshold_ : float
        The threshold value used for feature selection

    max_features_ : int
        Maximum number of features for feature selection

    use_cols_ : list of str
        Columns to select

    """

    def __init__(self, estimator, min_features=None, step=1, cv=None):

        self.estimator = estimator
        self.min_features = min_features
        self.step = step
        self.cv = cv



    def fit(self, X, y, groups=None):

        self.use_cols_ = list(X.columns)
        self.n_features_ = len(self.use_cols_)

        while True:

            if self.cv is None:
                imp = self._fit_importance(X[self.use_cols_], y)
            else:
                imp = self._cv_fit_importance(X[self.use_cols_], y, groups)

            self.use_cols_ = _select_k_best(imp, self.step, self.min_features)
            self.n_features_ = len(self.use_cols_)

            if self.n_features_ <= self.min_features:
                break

        return self



    def _fit_importance(self, X, y):

        estimator = clone(self.estimator).fit(X, y)
        imp = extract_importance(estimator)

        return imp



    def _cv_fit_importance(self, X, y, groups=None):

        cv = check_cv(self.cv, y, is_classifier(self.estimator_))
        imps = []

        for trn, _ in cv.split(X, y, groups):
            X_trn, y_trn = X.iloc[trn], y.iloc[trn]

            estimator = clone(self.estimator).fit(X_trn, y_trn)
            imp = extract_importance(estimator)
            imps.append(imp)

        imp = pd.concat(imps, axis=1).mean(axis=1)
        return imp



    def _select_features(self):
        return self.use_cols_



def _select_k_best(scores, step, min_features):

    n_features = len(scores)
    step = _check_step(step, n_features)
    k_best = max(n_features - step, min_features)

    sort_scores = scores.sort_values(ascending=False)
    best_scores = sort_scores.iloc[:k_best]

    use_cols = list(best_scores.index)
    return use_cols



def _check_step(step, n_features):

    if isinstance(step, int):
        if step > 0:
            step = step
        else:
            raise ValueError('Integer <step> must be greater than 0')

    elif isinstance(step, float):
        if 0 < step < 1:
            step = max(1, step * n_features)
        else:
            raise ValueError('Float <step> must be from interval (0, 1)')

    else:
        raise ValueError('Parameter <step> must be int or float, got {}'.format(step))

    return step
