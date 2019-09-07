from joblib import Parallel, delayed

from sklearn.utils.random import check_random_state
from sklearn.metrics.scorer import check_scoring

from tqdm import tqdm_notebook

import pandas as pd
import numpy as np


# Original: sklearn.inspection.permutation_importance.py

def _calculate_permutation_scores(estimator, X, y, col_idx, random_state,
                                  n_repeats, scorer):
    """Calculate score when `col_idx` is permuted."""

    x = X.iloc[:, col_idx].copy()
    scores = np.zeros(n_repeats)

    for n_round in range(n_repeats):

        X.iloc[:, col_idx] = random_state.permutation(x)
        X.iloc[:, col_idx] = X.iloc[:, col_idx].astype(x.dtype)

        scores[n_round] = scorer(estimator, X, y)

    return scores




def permutation_importance(estimator, X, y, scoring=None, n_repeats=5,
                           n_jobs=-1, random_state=0, progress_bar=False):
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

    n_repeats : int, default=5
        Number of times to permute a feature.

    n_jobs : int or None, default=None
        The number of jobs to use for the computation.
        `None` means 1. `-1` means using all processors.

    random_state : int, RandomState instance, or None, default=None
        Pseudo-random number generator to control the permutations of each
        feature.

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

    """

    random_state = check_random_state(random_state)
    scorer = check_scoring(estimator, scoring=scoring)

    baseline_score = scorer(estimator, X, y)
    scores = np.zeros((X.shape[1], n_repeats))


    parallel = Parallel(n_jobs=n_jobs, max_nbytes='256M') # FIXME: avoid <max_nbytes>

    scores = parallel(delayed(_calculate_permutation_scores)(estimator, X, y, col_idx,
        random_state, n_repeats, scorer) for col_idx in tqdm_notebook(range(X.shape[1])))


    importances = baseline_score - np.array(scores)

    result = {'importances_mean': np.mean(importances, axis=1),
              'importances_std': np.std(importances, axis=1),
              'importances': importances}

    return result
