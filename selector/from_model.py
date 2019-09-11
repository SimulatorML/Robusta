import pandas as pd
import numpy as np

from sklearn.exceptions import NotFittedError

from robusta.importance import extract_importance
from robusta.crossval import crossval

from .base import Selector


__all__ = ['SelectFromModel']

# Original: sklearn.feature_selection.SelectFromModel



class SelectFromModel(Selector):
    """Meta-transformer for selecting features based on importance weights.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        This can be both a fitted (if cv='prefit') or a non-fitted estimator.
        The estimator must have either a <feature_importances_> or <coef_>
        attribute after fitting.

    threshold : string, float, optional default None
        The threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded. If "median" (resp. "mean"), then the <threshold> value is
        the median (resp. the mean) of the feature importances. A scaling
        factor (e.g., "1.25*mean") may also be used. If None, drop features
        only based on <max_features>.

    max_features : int or None, optional
        The maximum number of features selected scoring above <threshold>.
        To disable <threshold> and only select based on <max_features>,
        set <threshold> to -np.inf.

    cv : int, cross-validation generator, iterable or "prefit"
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

            - None, to disable cross-validation and train single estimator
            on whole dataset.
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.
            - "prefit" string constant (default).

        If "prefit" is passed, it is assumed that <estimator> has been
        fitted already and <fit> function will raise error.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel (during cv). None means 1.

    verbose : int (default=1)
        Verbosity level (during cv)

    Attributes
    ----------
    estimator_ : list of fitted estimators, or single fitted estimator
        If <cv> is 'prefit'. If <cv> is None, return single estimator.
        Otherwise return list of fitted estimators, length (n_folds, ).

    feature_importances_ : Series of shape (n_features, )
        Feature importances, extracted from estimator(s)

    threshold_ : float
        The threshold value used for feature selection.

    use_cols_ : list of str
        Columns to select

    """

    def __init__(self, estimator, threshold=None, max_features=None,
                 cv='prefit', n_jobs=None, verbose=0):

        self.estimator = estimator
        self.threshold = threshold
        self.max_features = max_features
        self.cv = cv

        self.verbose = verbose
        self.n_jobs = n_jobs


    def fit(self, X, y, groups=None):

        # Check Cross-Validation
        if self.cv is 'prefit':
            raise NotFittedError("Since 'cv=prefit', call transform directly")

        elif self.cv is None:
            self.estimator_ = clone(self.estimator).fit(X_trn, y_trn)

        else:
            cv_result = crossval(self.estimator, self.cv, X, y, groups,
                                 return_pred=False, return_estimator=True,
                                 return_score=False, return_importance=False,
                                 return_encoder=False, return_folds=False,
                                 n_jobs=self.n_jobs, verbose=self.verbose):

            self.estimator_ = cv_result['estimator']

        return self


    def _fit(self, X, y, groups=None):

        self.estimator_ = clone(self.estimator).fit(X_trn, y_trn)

        return self

        # Check Cross-Validation
        if self.cv is None:
            ii = np.arange(X.shape[0]) # full dataset
            cv = np.array([(ii, ii)])
        else:
            cv = self.cv

        cv = check_cv(cv, y, classifier=is_classifier(self.estimator))

        # Fit & calculate importances
        self.estimator_ = []
        imps = []

        for trn, oof in cv.split(X, y, groups):

            X_trn, y_trn = X.iloc[trn], y.iloc[trn]
            X_oof, y_oof = X.iloc[oof], y.iloc[oof]

            estimator = clone(self.estimator).fit(X_trn, y_trn)

            self.estimators_.append(estimator)

        return self


    def _cv_fit(self, X, y, groups=None):

        # Check Cross-Validation
        if self.cv is None:
            ii = np.arange(X.shape[0]) # full dataset
            cv = np.array([(ii, ii)])
        else:
            cv = self.cv

        cv = check_cv(cv, y, classifier=is_classifier(self.estimator))

        # Fit & calculate importances
        self.estimators_ = []
        imps = []

        for trn, oof in cv.split(X, y, groups):

            X_trn, y_trn = X.iloc[trn], y.iloc[trn]
            X_oof, y_oof = X.iloc[oof], y.iloc[oof]

            if self.cv is 'prefit':
                estimator = self.estimator
            else:
                estimator = clone(self.estimator).fit(X_trn, y_trn)

            self.estimators_.append(estimator)

            imp = extract_importance(estimator)
            imps.append(imp)

        imp = pd.concat(imps, axis=1).mean(axis=1)
        self.feature_importances_ = imp

        return self



    def _extract_importance(self):

        imps = []

        if self.cv is 'prefit':
            estimators = [self.estimator]

        elif self.cv is None:
            estimators = [self.estimator_]

        else:
            estimators = self.estimator_

        for estimator in estimators:
            imp = extract_importance(estimator)
            imps.append(imp)

        imp = pd.concat(imps, axis=1).mean(axis=1)
        self.feature_importances_ = imp

        return imp


    def _select_features(self):

        imp = self._extract_importance()

        if isinstance(self.threshold, np.number)
        or isinstance(self.threshold, 'str'):
            self.threshold_ = _calculate_threshold(imp, self.threshold)
            threshold_mask = (imp >= self.threshold_)

        if isinstance(self.max_features, np.number):
            ranking = imp.rank(ascending=False)
            ranking_mask = (ranks <= self.max_features)

        use_cols = imp.index[threshold_mask & ranking_mask]
        return use_cols




def _calculate_threshold(importances, threshold):
    """Interpret the threshold value"""

    if threshold is None:
        threshold = -np.inf

    if isinstance(threshold, str):
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
