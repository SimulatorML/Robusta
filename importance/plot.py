import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plot_importance(imps, features=None, k_top=30):

    imps = [pd.Series(imp) for imp in imps]
    imp = pd.concat(imps, axis=1)
    imp_mean = imp.mean(axis=1)
    imp_sort = imp_mean.sort_values(ascending=False)
    features = list(imp_sort[:k_top].index) if not features else features

    imp_vals = [x.values for x in imps]
    imp_inds = [x.index for x in imps]
    imp_vals = np.hstack(imp_vals)
    imp_inds = np.hstack(imp_inds)
    imp = pd.Series(imp_vals, index=imp_inds)
    imp.index.name = 'feature'
    imp.name = 'importance'

    sns.barplot(x='importance', y='feature', data=imp.loc[features].reset_index(),
                edgecolor=('#d4c3a3'), linewidth=2, palette="inferno_r")
    plt.grid(False)
    plt.show()
