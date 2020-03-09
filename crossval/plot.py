import pandas as pd
import numpy as np

from collections import defaultdict

from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.metrics import check_scoring, roc_curve
from sklearn.model_selection import check_cv

from joblib import Parallel, delayed
from scipy import interp, stats

import matplotlib.pyplot as plt
import seaborn as sns

from .results import check_cvs
from ._curve import *


__all__ = [
    'plot_learing_curve',
    'plot_roc_auc',
    'plot_ttest',
]




def plot_learing_curve(result, X, y, groups=None, max_iter=0, step=1,
                       mode='mean', train_score=False, n_jobs=None):

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

    mode : {'mean', 'fold', 'both'} (default='mean')
        - 'mean' : plot average score and std (default)
        - 'fold' : plot score of each fold
        - 'both' : plot both

    train_score : bool (default=False)
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

    modes = ('mean', 'fold', 'both')
    assert mode in modes, f'<mode> must be from {modes}. Found {mode}'

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

    if not train_score:
        trn_scores = None
    else:
        avg = trn_scores.mean(axis=0)
        std = trn_scores.std(axis=0)

        if mode in ['mean', 'both']:
            plt.fill_between(stages, avg-std, avg+std, alpha=.1, color='b')
            plt.plot(stages, avg, label='train score', color='b')

        if mode in ['fold', 'both']:
            for scores in trn_scores:
                plt.plot(stages, scores, '--', color='b', lw=0.5, alpha=0.5)

    if True:
        avg = val_scores.mean(axis=0)
        std = val_scores.std(axis=0)

        if mode in ['mean', 'both']:
            plt.fill_between(stages, avg-std, avg+std, alpha=.1, color='y')
            plt.plot(stages, avg, label='valid score', color='y')

        if mode in ['fold', 'both']:
            for scores in val_scores:
                plt.plot(stages, scores, '--', color='y', lw=0.5, alpha=0.5)

    plt.legend()
    plt.show()

    return trn_scores, val_scores



def plot_ttest(resultA, resultB, score='val_score', label='label', cuped=False):

    # Check input
    assert score in resultA, f"<resultA> has no '{key}'"
    assert score in resultB, f"<resultB> has no '{key}'"
    a = resultA[score]
    b = resultB[score]

    assert len(a) == len(b), 'Both scores must be of the same size'
    n = len(a)

    # Check labels
    labels = ['0', '1']
    if label in resultA: labels[0] = resultA[label]
    if label in resultB: labels[1] = resultB[label]

    # CUPED
    if cuped:
        theta = np.cov(a, b)[0, 1] / np.var(a)
        b -= (a - np.mean(b)) * theta

    # t-test
    t, p = stats.ttest_rel(a, b)

    # Plot
    _, axes = plt.subplots(2, 2)

    # Plot box
    ax = axes[0, 0]
    sns.boxplot(labels, [a, b], linewidth=2.0, ax=ax)
    ax.grid(alpha=0.2)

    # Plot pairs
    ax = axes[1, 0]
    for x, y in zip(a, b):
        ax.plot(labels, [x, y], 'o-', color='b', alpha=0.8)
    ax.plot(labels, [np.mean(a), np.mean(b)], 'o-', color='w')
    ax.grid(alpha=0.2)

    # Plot dist
    ax = axes[0, 1]
    sns.distplot(a, 10, label=labels[0], ax=ax)
    sns.distplot(b, 10, label=labels[1], ax=ax)
    ax.grid(alpha=0.2)
    ax.legend()

    # Plot proba
    ax = axes[1, 1]
    x_abs = max(5, abs(t))
    x_min, x_max = -x_abs, +x_abs

    xx = np.arange(t, x_max, 0.001)
    yy = stats.t.pdf(xx, n-1)
    ax.plot(xx, yy, color='gray')
    ax.fill_between(xx, yy, color='gray', alpha=0.2)

    xx = np.arange(x_min, t, 0.001)
    yy = stats.t.pdf(xx, n-1)
    ax.plot(xx, yy, color='r')
    ax.fill_between(xx, yy, color='r', alpha=0.2)

    ax.legend(['t-value = {:.4f}'.format(t),
               'p-value = {:.4f}'.format(p)])
    ax.grid(alpha=0.2)

    return t, p




def plot_roc_auc(results, X, y, groups=None, labels=None, colors=None, steps=200):

    # Check input
    cv = check_cvs(results, X, y, groups)

    msg = "<labels> must be of same len as <results>"
    if labels:
        assert len(labels) == len(results), msg
    else:
        labels = list(range(len(results)))

    msg = "<colors> must be of same len as <results>"
    if colors:
        assert len(colors) == len(results), msg
    else:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    msg = "Each <result> must have 'estimator' key"
    for result in results:
        assert 'estimator' in result, msg

    # Get curves
    avg_fpr = np.linspace(0, 1, steps)
    curves = defaultdict(list)
    cv = results[0]['cv']

    for i, (_, oof) in enumerate(cv.split(X, y, groups)):

        X_oof = X.iloc[oof]
        y_oof = y.iloc[oof]

        for j, result in enumerate(results):
            y_pred = result['estimator'][i].predict_proba(X_oof)[:, 1]
            fpr, tpr, _ = roc_curve(y_oof, y_pred)
            tpr = interp(avg_fpr, fpr, tpr)
            tpr[0] = 0.0
            curves[labels[j]].append(tpr)

    # Plot
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = dict(zip(labels, colors))

    plt.figure()
    for label, tprs in curves.items():
        c = colors[label]

        for tpr in tprs:
            plt.plot(avg_fpr, tpr, c=c, alpha=0.2)

        avg_tpr = np.mean(tprs, axis=0)
        plt.plot(avg_fpr, avg_tpr, c=c, label=label)

        std_tpr = np.std(tprs, axis=0)
        tpr_upper = np.minimum(avg_tpr + std_tpr, 1)
        tpr_lower = np.maximum(avg_tpr - std_tpr, 0)
        plt.fill_between(avg_fpr, tpr_lower, tpr_upper, color=c, alpha=.1)

    plt.legend(loc='lower right')
    plt.show()
