import pandas as pd
import numpy as np

from sklearn.model_selection import check_cv
from sklearn.exceptions import NotFittedError
from sklearn.base import clone, is_classifier

from robusta.importance import get_importance
from robusta.crossval import crossval

from .base import _Selector

# Original: sklearn.feature_selection.SelectFromModel



class SelectFromModel(_Selector):
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

    def __init__(self, estimator, cv=None, threshold=None, max_features=None):

        self.estimator = estimator
        self.threshold = threshold
        self.max_features = max_features
        self.cv = cv


    def fit(self, X, y, groups=None):

        if self.cv is 'prefit':
            raise NotFittedError("Since 'cv=prefit', call transform directly")

        elif self.cv is None:
            self.estimator_ = clone(self.estimator).fit(X, y)

        else:
            self.estimator_ = []
            cv = check_cv(self.cv, y, is_classifier(self.estimator_))

            for trn, _ in cv.split(X, y, groups):
                X_trn, y_trn = X.iloc[trn], y.iloc[trn]

                estimator = clone(self.estimator).fit(X_trn, y_trn)
                self.estimator_.append(estimator)

        return self


    @property
    def feature_importances_(self):

        imps = []

        if self.cv is 'prefit':
            estimators = [self.estimator]
        elif self.cv is None:
            estimators = [self.estimator_]
        else:
            estimators = self.estimator_

        for estimator in estimators:
            imp = get_importance(estimator)
            imps.append(imp)

        return pd.concat(imps, axis=1).mean(axis=1)


    def get_features(self):

        imp = self.feature_importances_

        self.threshold_ = _check_threshold(imp, self.threshold)
        threshold_mask = (imp >= self.threshold_)

        self.max_features_ = _check_max_features(imp, self.max_features)
        ranking_mask = (imp.rank(ascending=False) <= self.max_features_)

        use_cols = imp.index[threshold_mask & ranking_mask]
        return list(use_cols)



def _check_max_features(importances, max_features):
    """Interpret the max_features value"""

    n_features = len(importances)

    if max_features is None:
        max_features = n_features

    elif isinstance(max_features, int):
        max_features = min(n_features, max_features)

    elif isinstance(max_features, float):
        max_features = int(n_features * max_features)

    return max_features



def _check_threshold(importances, threshold):
    """Interpret the threshold value"""

    if threshold is None:
        threshold = -np.inf

    elif isinstance(threshold, str):
        if "*" in threshold:
            scale, reference = threshold.split("*")
            scale = float(scale.strip())
            reference = reference.strip()

            if reference == "median":
                reference = np.median(importances)
            elif reference == "mean":
                reference = np.mean(importances)
            else:
                raise ValueError("Unknown reference: " + reference)

            threshold = scale * reference

        elif threshold == "median":
            threshold = np.median(importances)

        elif threshold == "mean":
            threshold = np.mean(importances)

        else:
            raise ValueError("Expected threshold='mean' or threshold='median' "
                             "got %s" % threshold)

    else:
        threshold = float(threshold)

    return threshold
