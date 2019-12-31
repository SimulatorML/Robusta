import pandas as pd
import numpy as np

from scipy.stats import ttest_rel, ttest_ind, norm

__all__ = [
    'compare_ttest',
]


def compare_ttest(resultA, resultB, key='val_score', ind=True):

    # Check input
    assert key in resultA, f"<resultA> has no '{key}'"
    assert key in resultB, f"<resultB> has no '{key}'"
    a = resultA[key]
    b = resultB[key]

    # t-test
    t, p = ttest_rel(a, b)

    # Plot
    _, axes = plt.subplots(2, 2)

    # Plot box
    ax = axes[0, 0]
    sns.boxplot(['A', 'B'], [a, b], linewidth=2.0, ax=ax)
    ax.grid(alpha=0.2)

    # Plot pairs
    ax = axes[1, 0]
    for x, y in zip(a, b):
        ax.plot(['A', 'B'], [x, y], 'o-', color='b', alpha=0.8)
    ax.plot(['A', 'B'], [np.mean(a), np.mean(b)], 'o-', color='w')
    ax.grid(alpha=0.2)

    # Plot dist
    ax = axes[0, 1]
    sns.distplot(a, 10, label='A', ax=ax)
    sns.distplot(b, 10, label='B', ax=ax)
    ax.grid(alpha=0.2)
    ax.legend()

    # Plot proba
    ax = axes[1, 1]

    xx = np.arange(-abs(t), abs(t), 0.001)
    yy = norm.pdf(xx, 0, 1)
    ax.plot(xx, yy, color='gray')
    ax.fill_between(xx, yy, color='gray', alpha=0.2)

    xx = np.arange(-5, t, 0.001)
    yy = norm.pdf(xx, 0, 1)
    ax.plot(xx, yy, color='r')
    ax.fill_between(xx, yy, color='r', alpha=0.2)

    xx = np.arange(abs(t), 5, 0.001)
    yy = norm.pdf(xx, 0, 1)
    ax.plot(xx, yy, color='r')
    ax.fill_between(xx, yy, color='r', alpha=0.2)

    ax.legend(['t-value = {:.4f}'.format(t),
               'p-value = {:.4f}'.format(p)])
    ax.grid(alpha=0.2)

    return t, p
