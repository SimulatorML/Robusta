import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

from ._curve import *


__all__ = [
    'plot_learing_curve',
]




def plot_learing_curve(result, X, y, groups=None, max_iter=0, step=1,
                       train_score=True, n_jobs=None):

    """Plot learning curve for boosting estimators.

    Currently supported:
        - LGBMClassifier, LGBMRegressor
        - CatBoostClassifier, CatBoostRegressor

    Parameters
    ----------
    result : dict
        Cross-validation results, returned by <crossval> function.
        Must contain 'estimator', 'scorer' and 'cv' keys.

    X : DataFrame, shape [n_samples, n_features]
        The data to fit, score and calculate out-of-fold predictions.
        Must be the same as during cross-validation fit.

    y : Series, shape [n_samples]
        The target variable to try to predict.
        Must be the same as during cross-validation fit.

    groups : None
        Group labels for the samples used while splitting the dataset into
        train/test set.
        Must be the same as during cross-validation fit.

    max_iter : int (default=0)
        Maximum number of trees. 0 means all.

    step : int (default=1)
        If greater than 1, plot score only for trees with indices:
        step-1, 2*step-1, 3*step-1 & etc (zero-based indices).
        Larger step speeds up prediction.

    train_score : bool (default=True)
        Whether to plot learning curve for training scores.
        If False, speeds up prediction.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.
        

    Returns
    -------
    trn_scores : ndarray, shape (n_folds, n_stages)
        Train scores learning curve for each fold.
        If train_score is False, return None.

    val_scores : ndarray, shape (n_folds, n_stages)
        Validation scores learning curve for each fold.

    """

    estimators = result['estimator']
    scorer = result['scorer']
    cv = result['cv']

    # Estimator Name
    estimator = estimators[0]
    name = estimator.__class__.__name__

    if name.startswith('CatBoost'):
        generator = _cat_staged_predict
        if max_iter == 0:
            max_iter = min([e.tree_count_ for e in estimators])

    elif name.startswith('LGB'):
        generator = _lgb_staged_predict
        if max_iter == 0:
            max_iter = min([e.booster_.num_trees() for e in estimators])

    elif name.startswith('XGB'):
        raise NotImplementedError('XGBoost currently does not supported')
        generator = _xgb_staged_predict
        if max_iter == 0:
            max_iter = min([e.n_estimators for e in estimators])

    else:
        raise NotImplementedError('Only LGBM and CatBoost currently supported')

    # Estimator Type
    if estimator._estimator_type == 'classifier':
        predictor = _StagedClassifier()

    elif estimator._estimator_type == 'regressor':
        predictor = _StagedRegressor()

    # Predict in Parallel
    stages = np.arange(step, max_iter+step, step)
    folds = cv.split(X, y, groups)

    scores = Parallel(n_jobs=n_jobs)(
        delayed(_get_scores)(estimator, generator, predictor, trn, val, X, y,
                             scorer, max_iter, step, train_score)
        for (trn, val), estimator in zip(folds, estimators)
    )

    trn_scores = np.array([s[0] for s in scores])
    val_scores = np.array([s[1] for s in scores])

    # Learning Curve(s)
    plt.figure()

    if train_score:
        trn_avg = trn_scores.mean(axis=0)
        trn_std = trn_scores.std(axis=0)

        plt.fill_between(stages, trn_avg-trn_std, trn_avg+trn_std, alpha=.1, color='b')
        plt.plot(stages, trn_scores.mean(axis=0), label='train score', color='b')

    if True:
        val_avg = val_scores.mean(axis=0)
        val_std = val_scores.std(axis=0)

        plt.fill_between(stages, val_avg-val_std, val_avg+val_std, alpha=.1, color='y')
        plt.plot(stages, val_scores.mean(axis=0), label='valid score', color='y')

    plt.legend()
    plt.show()

    if train_score:
        return trn_scores, val_scores
    else:
        return None, val_scores
