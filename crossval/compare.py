import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve
from collections import defaultdict
from scipy import interp, stats

import matplotlib.pyplot as plt
import seaborn as sns

from .results import check_cvs


__all__ = [
    'compare_roc_auc',
    'compare_ttest',
]




def compare_ttest(resultA, resultB, labels=['A', 'B'], key='val_score'):

    # Check input
    assert key in resultA, f"<resultA> has no '{key}'"
    assert key in resultB, f"<resultB> has no '{key}'"
    a = resultA[key]
    b = resultB[key]
    assert len(a) == len(b), 'Both scores must be of the same size'
    n = len(a)

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




def compare_roc_auc(results, X, y, groups=None, labels=None, colors=None,
                    steps=200):

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
