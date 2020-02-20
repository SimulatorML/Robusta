from joblib import Parallel, delayed

from tqdm import tqdm_notebook

import pandas as pd
import numpy as np

from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.random import check_random_state
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    is_classifier,
)



# Original:
#
# sklearn.inspection.permutation_importance.py
# https://scikit-learn.org/dev/modules/generated/sklearn.inspection.permutation_importance.html
#
# eli5.sklearn.permutation_importance
# https://eli5.readthedocs.io/en/latest/_modules/eli5/sklearn/permutation_importance.html


def _get_col_score(estimator, X, y, col, n_repeats, scorer, rstate):
    """Calculate score when `col` is permuted."""

    x = X.loc[:, col].copy()
    scores = np.zeros(n_repeats)

    for i in range(n_repeats):

        X.loc[:, col] = rstate.permutation(x)
        X.loc[:, col] = X.loc[:, col].astype(x.dtype)

        score = scorer(estimator, X, y) # bottleneck
        scores[i] = score

    return scores



def get_col_score(estimator, X, y, col, n_repeats=5, scoring=None,
                  random_state=None):
    """Calculate score when `col` is permuted."""

    scorer = check_scoring(estimator, scoring=scoring)
    rstate = check_random_state(random_state)

    scores = _get_col_score(estimator, X, y, col, n_repeats, scorer, rstate)

    return scores



def _get_group_score(estimator, X, y, g, n_repeats, scorer, rstate):
    """Calculate score when columns group `g` is permuted."""

    x = X.loc[:, g].copy()
    scores = np.zeros(n_repeats)

    for i in range(n_repeats):

        X.loc[:, g] = rstate.permutation(x)
        X.loc[:, g] = X.loc[:, g].astype(x.dtypes)

        score = scorer(estimator, X, y) # bottleneck
        scores[i] = score

    return scores



def get_group_score(estimator, X, y, g, n_repeats=5, scoring=None,
                    random_state=None):
    """Calculate score when columns group `g` is permuted."""

    scorer = check_scoring(estimator, scoring=scoring)
    rstate = check_random_state(random_state)

    scores = _get_group_score(estimator, X, y, g, n_repeats, scorer, rstate)

    return scores



def permutation_importance(estimator, X, y, subset=None, scoring=None, n_repeats=5,
                           n_jobs=None, random_state=0, tqdm=False):
    """Permutation importance for feature evaluation [BRE].

    The 'estimator' is required to be a fitted estimator. 'X' can be the
    data set used to train the estimator or a hold-out set. The permutation
    importance of a feature is calculated as follows. First, a baseline metric,
    defined by 'scoring', is evaluated on a (potentially different) dataset
    defined by the 'X'. Next, a feature column from the validation set is
    permuted and the metric is evaluated again. The permutation importance
    is defined to be the difference between the baseline metric and metric
    from permutating the feature column.

    Original: sklearn.inspection.permutation_importance.py

    Parameters
    ----------
    estimator : object
        An estimator that has already been fitted and is compatible
        with 'scorer'.

    X : DataFrame, shape (n_samples, n_features)
        Data on which permutation importance will be computed.

    y : array-like or None, shape (n_samples, ) or (n_samples, n_classes)
        Targets for supervised or `None` for unsupervised.

    subset : list or None (default=None)
        If not None, score only subset of features.
        Other importances will be NaN.

    scoring : string, callable or None, default=None
        Scorer to use. It can be a single string or a callable.
        If None, the estimator's default scorer is used.

    n_repeats : int, default=5
        Number of times to permute a feature.

    n_jobs : int or None, default=None
        The number of jobs to use for the computation.
        `None` means 1. `-1` means using all processors.

    random_state : int, RandomState instance, or None, default=None
        Pseudo-random number generator to control the permutations of each
        feature.

    tqdm : bool, default=False
        Whether to display <tqdm_notebook> progress bar while iterating
        through out dataset columns.

    Returns
    -------
    result : Bunch
        Dictionary-like object, with attributes:

        importances_mean : ndarray, shape (n_features, )
            Mean of feature importance over `n_repeats`.

        importances_std : ndarray, shape (n_features, )
            Standard deviation over `n_repeats`.

        importances : ndarray, shape (n_features, n_repeats)
            Raw permutation importance scores.

        score : float
            Baseline score

    """

    columns = list(X.columns)

    msg = "<subset> must contain only features from <X>"
    assert not subset or not set(subset) - set(columns), msg

    msg = "<subset> must contain only unique features"
    assert not subset or len(set(subset)) == len(subset), msg

    subset = subset if subset else columns
    subset = tqdm_notebook(subset) if tqdm else subset

    scorer = check_scoring(estimator, scoring=scoring)
    rstate = check_random_state(random_state)

    base_score = scorer(estimator, X, y)
    scores = np.zeros((len(subset), n_repeats))

    # FIXME: avoid <max_nbytes>
    scores = Parallel(n_jobs=n_jobs, max_nbytes='512M', backend='multiprocessing')(
        delayed(_get_col_score)(estimator, X, y, feature, n_repeats, scorer, rstate)
        for feature in subset)

    importances = np.full((len(columns), n_repeats), np.nan)
    ind = [columns.index(feature) for feature in subset]
    importances[ind] = base_score - np.array(scores)

    result = {'importances_mean': np.mean(importances, axis=1),
              'importances_std': np.std(importances, axis=1),
              'importances': importances,
              'score': base_score}

    return result



def group_permutation_importance(estimator, X, y, subset=None, scoring=None,
                                 n_repeats=5, n_jobs=None, random_state=0,
                                 tqdm=False):

    columns = list(X.columns.get_level_values(0).unique())

    msg = "<subset> must contain only features from <X>"
    assert not subset or not set(subset) - set(columns), msg

    msg = "<subset> must contain only unique features"
    assert not subset or len(set(subset)) == len(subset), msg

    subset = subset if subset else columns
    subset = tqdm_notebook(subset) if tqdm else subset

    scorer = check_scoring(estimator, scoring=scoring)
    rstate = check_random_state(random_state)

    base_score = scorer(estimator, X, y)
    scores = np.zeros((len(subset), n_repeats))

    # FIXME: avoid <max_nbytes>
    scores = Parallel(n_jobs=n_jobs, max_nbytes='512M', backend='multiprocessing')(
        delayed(_get_group_score)(estimator, X, y, feature, n_repeats, scorer, rstate)
        for feature in subset)

    importances = np.full((len(columns), n_repeats), np.nan)
    ind = [columns.index(feature) for feature in subset]
    importances[ind] = base_score - np.array(scores)

    result = {'importances_mean': np.mean(importances, axis=1),
              'importances_std': np.std(importances, axis=1),
              'importances': importances,
              'score': base_score}

    return result





class PermutationImportance(BaseEstimator, MetaEstimatorMixin):
    """Meta-estimator which computes ``feature_importances_`` attribute
    based on permutation importance (also known as mean score decrease).

    Parameters
    ----------
    estimator : object
        The base estimator. This can be both a fitted
        (if ``prefit`` is set to True) or a non-fitted estimator.

    scoring : string, callable or None, default=None
        Scoring function to use for computing feature importances.
        A string with scoring name (see scikit-learn docs) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    cv : int, cross-validation generator, iterable or "prefit"
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

            - None, to disable cross-validation and compute feature importances
              on the same data as used for training.
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.
            - "prefit" string constant (default).

        If "prefit" is passed, it is assumed that ``estimator`` has been
        fitted already and all data is used for computing feature importances.

    n_repeats : int, default 5
        The number of random shuffle iterations. Decrease to improve speed,
        increase to get more precise estimates.

    random_state : integer or numpy.random.RandomState, optional
        Pseudo-random number generator to control the permutations of each feature.

    tqdm : bool, default=False
        Weather to display <tqdm_notebook> progress bar while iterating
        through out dataset columns.

    n_jobs : int, default -1
        The number of jobs to run in parallel. None means 1.

    y_transform : callable, default=None
        Transform target before fit

    Attributes
    ----------
    feature_importances_ : Series, shape (n_groups, )
        Feature importances, computed as mean decrease of the score when
        a feature is permuted (i.e. becomes noise).

    feature_importances_std_ : Series, shape (n_groups, )
        Standard deviations of feature importances.

    raw_importances_ : list of Dataframes, shape (n_folds, n_groups, n_repeats)

    scores_ : list of floats, shape (n_folds, )

    """
    def __init__(self, estimator, cv='prefit', scoring=None, n_repeats=5,
                 random_state=None, tqdm=False, n_jobs=None, y_transform=None):

        self.estimator = estimator
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.cv = cv

        self.random_state = random_state
        self.tqdm = tqdm
        self.n_jobs = n_jobs

        self.y_transform = y_transform


    def fit(self, X, y, groups=None, subset=None, **fit_params):
        """Compute ``feature_importances_`` attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        subset : list or None (default=None)
            If not None, score only subset of features.
            Other importances will be NaN.

        **fit_params : Other estimator specific parameters

        Returns
        -------
        self : object
            Returns self.

        """

        if self.cv in ['prefit', None]:
            ii = np.arange(X.shape[0]) # full dataset
            cv = np.array([(ii, ii)])
        else:
            cv = self.cv

        cv = check_cv(cv, y, classifier=is_classifier(self.estimator))

        self.raw_importances_ = []
        self.scores_ = []

        for trn, oof in cv.split(X, y, groups):

            X_trn, y_trn = X.iloc[trn], y.iloc[trn]
            X_oof, y_oof = X.iloc[oof], y.iloc[oof]

            y_trn_ = self.y_transform(y_trn) if self.y_transform else y_trn

            if self.cv is 'prefit':
                estimator = self.estimator
            else:
                estimator = clone(self.estimator).fit(X_trn, y_trn_, **fit_params)

            pi = permutation_importance(estimator, X_oof, y_oof, subset=subset,
                                        n_repeats=self.n_repeats,
                                        scoring=self.scoring,
                                        random_state=self.random_state,
                                        tqdm=self.tqdm, n_jobs=self.n_jobs)

            imp = pd.DataFrame(pi['importances'], index=X.columns)
            self.raw_importances_.append(imp)
            self.scores_.append(pi['score'])

        imps = pd.concat(self.raw_importances_, axis=1)

        self.feature_importances_ = imps.mean(axis=1)
        self.feature_importances_std_ = imps.std(axis=1)

        return self

    # ============= Exposed methods of a wrapped estimator:

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def score(self, X, y=None, *args, **kwargs):
        return self.wrapped_estimator_.score(X, y, *args, **kwargs)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def predict(self, X):
        return self.wrapped_estimator_.predict(X)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def predict_proba(self, X):
        return self.wrapped_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def predict_log_proba(self, X):
        return self.wrapped_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def decision_function(self, X):
        return self.wrapped_estimator_.decision_function(X)

    @property
    def wrapped_estimator_(self):
        if self.cv == "prefit" or not self.refit: return self.estimator
        return self.estimator_

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def classes_(self):
        return self.wrapped_estimator_.classes_




class GroupPermutationImportance(PermutationImportance):

    def fit(self, X, y, groups=None, subset=None, **fit_params):
        """Compute ``feature_importances_`` attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        **fit_params : Other estimator specific parameters

        Returns
        -------
        self : object
            Returns self.

        """

        col_group = X.columns.get_level_values(0).unique()

        if self.cv in ['prefit', None]:
            ii = np.arange(X.shape[0]) # full dataset
            cv = np.array([(ii, ii)])
        else:
            cv = self.cv

        cv = check_cv(cv, y, classifier=is_classifier(self.estimator))

        self.raw_importances_ = []
        self.scores_ = []

        for trn, oof in cv.split(X, y, groups):

            X_trn, y_trn = X.iloc[trn], y.iloc[trn]
            X_oof, y_oof = X.iloc[oof], y.iloc[oof]

            y_trn_ = self.y_transform(y_trn) if self.y_transform else y_trn

            if self.cv is 'prefit':
                estimator = self.estimator
            else:
                estimator = clone(self.estimator).fit(X_trn, y_trn_, **fit_params)

            pi = group_permutation_importance(estimator, X_oof, y_oof, subset=subset,
                                              n_repeats=self.n_repeats,
                                              scoring=self.scoring,
                                              random_state=self.random_state,
                                              tqdm=self.tqdm, n_jobs=self.n_jobs)

            imp = pd.DataFrame(pi['importances'], index=col_group)

            self.raw_importances_.append(imp)
            self.scores_.append(pi['score'])

        imps = pd.concat(self.raw_importances_, axis=1)

        self.feature_importances_ = imps.mean(axis=1)
        self.feature_importances_std_ = imps.std(axis=1)

        return self
