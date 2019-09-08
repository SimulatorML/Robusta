from joblib import Parallel, delayed

from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.random import check_random_state

from sklearn.metrics.scorer import check_scoring

from tqdm import tqdm_notebook

import pandas as pd
import numpy as np

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


def _get_col_score(estimator, X, y, col, n_iter, scorer, rstate):
    """Calculate score when `col` is permuted."""

    dtype = X[col].dtype
    scores = np.zeros(n_iter)

    for i in range(n_iter):

        X[col] = rstate.permutation(X[col])
        X[col] = X[col].astype(dtype)

        score = scorer(estimator, X, y) # bottleneck
        scores[i] = score

    return scores



def get_col_score(estimator, X, y, col, n_iter=5, scoring=None, random_state=None):
    """Calculate score when `col` is permuted."""

    scorer = check_scoring(estimator, scoring=scoring)
    rstate = check_random_state(random_state)

    scores = _get_col_score(estimator, X, y, col, n_iter, scorer, rstate)

    return scores



def permutation_importance(estimator, X, y, scoring=None, n_iter=5, n_jobs=-1,
                           random_state=0, progress_bar=False):
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

    scoring : string, callable or None, default=None
        Scorer to use. It can be a single string or a callable.
        If None, the estimator's default scorer is used.

    n_iter : int, default=5
        Number of times to permute a feature.

    n_jobs : int or None, default=None
        The number of jobs to use for the computation.
        `None` means 1. `-1` means using all processors.

    random_state : int, RandomState instance, or None, default=None
        Pseudo-random number generator to control the permutations of each
        feature.

    progress_bar : bool, default=False
        Weather to display <tqdm_notebook> progress bar while iterating
        through out dataset columns.

    Returns
    -------
    result : Bunch
        Dictionary-like object, with attributes:

        importances_mean : ndarray, shape (n_features, )
            Mean of feature importance over `n_iter`.

        importances_std : ndarray, shape (n_features, )
            Standard deviation over `n_iter`.

        importances : ndarray, shape (n_features, n_iter)
            Raw permutation importance scores.

    """

    cols = tqdm_notebook(X.columns) if progress_bar else X.columns

    scorer = check_scoring(estimator, scoring=scoring)
    rstate = check_random_state(random_state)

    baseline_score = scorer(estimator, X, y)
    scores = np.zeros((len(cols), n_iter))

    # FIXME: avoid <max_nbytes>
    scores = Parallel(n_jobs=n_jobs, max_nbytes='256M')(
        delayed(_get_col_score)(estimator, X, y, col, n_iter, scorer, rstate)
        for col in cols)

    importances = baseline_score - np.array(scores)

    result = {'importances_mean': np.mean(importances, axis=1),
              'importances_std': np.std(importances, axis=1),
              'importances': importances}

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

    n_iter : int, default 5
        The number of random shuffle iterations. Decrease to improve speed,
        increase to get more precise estimates.

    n_jobs : int, default -1
        The number of jobs to run in parallel. None means 1.

    random_state : integer or numpy.random.RandomState, optional
        Pseudo-random number generator to control the permutations of each feature.

    progress_bar : bool, default=False
        Weather to display <tqdm_notebook> progress bar while iterating
        through out dataset columns.

    Attributes
    ----------
    feature_importances_ : Series
        Feature importances, computed as mean decrease of the score when
        a feature is permuted (i.e. becomes noise).

    feature_importances_std_ : Series
        Standard deviations of feature importances.

    scores_ : list of arrays
        A list of score decreases for all experiments.

    """
    def __init__(self, estimator, scoring=None, n_iter=5, n_jobs=-1,
                 random_state=None, progress_bar=False):

        self.estimator = estimator
        self.scoring = scoring
        self.n_iter = n_iter

        self.random_state = random_state
        self.progress_bar = progress_bar
        self.n_jobs = n_jobs


    def fit(self, X, y, groups=None, **fit_params):
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
        result = permutation_importance(self.estimator, X, y, n_iter=self.n_iter,
                                        scoring=self.scoring, n_jobs=self.n_jobs,
                                        random_state=self.random_state,
                                        progress_bar=self.progress_bar)

        fi_mean, fi_std = result['importances_mean'], result['importances_std']

        self.feature_importances_ = pd.Series(fi_mean, index=X.columns)
        self.feature_importances_std_ = pd.Series(fi_std, index=X.columns)

        fi = result['importances']

        self.scores_ = pd.DataFrame(fi, index=X.columns)

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
        if self.cv == "prefit" or not self.refit:
            return self.estimator
        return self.estimator_

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def classes_(self):
        return self.wrapped_estimator_.classes_
