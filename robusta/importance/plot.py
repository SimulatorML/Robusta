from typing import List, Optional

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plot_importance(
    imps: List[pd.Series],
    features: Optional[List[str]] = None,
    k_top: Optional[int] = 30,
) -> None:
    """
    Plot feature importance using barplot.

    Parameters
    ----------
    imps : List[pd.Series]
        List of pandas Series, each containing feature importances.
    features : Optional[List[str]]
        list of str, names of features to be plotted. If None, the top k_top features will be plotted.
    k_top : Optional[int]
        int, number of top features to plot.

    Returns
    -------
    Nothing :
        None.
    """
    # Calculate mean importance across all input importances
    imps = [pd.Series(imp) for imp in imps]
    imp = pd.concat(imps, axis=1)
    imp_mean = imp.mean(axis=1)

    # Sort features by mean importance and select top k_top features
    imp_sort = imp_mean.sort_values(ascending=False)
    features = list(imp_sort[:k_top].index) if not features else features

    # Merge all importances into a single pandas Series
    imp_vals = [x.values for x in imps]
    imp_inds = [x.index for x in imps]
    imp_vals = np.hstack(imp_vals)
    imp_inds = np.hstack(imp_inds)
    imp = pd.Series(imp_vals, index=imp_inds)
    imp.index.name = "feature"
    imp.name = "importance"

    # Plot feature importance barplot
    sns.barplot(
        x="importance",
        y="feature",
        data=imp.loc[features].reset_index(),
        edgecolor=("#d4c3a3"),
        linewidth=2,
        palette="inferno_r",
    )
    plt.grid(False)
    plt.show()
