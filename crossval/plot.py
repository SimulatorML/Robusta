import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring

import matplotlib.pyplot as plt

from ._curve import *


__all__ = [
    'plot_curve',
]




def plot_curve(result, X, y, groups=None, max_iter=0, step=1, n_jobs=None,
               train_score=True):

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
        return val_scores
