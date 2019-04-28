import os, shutil, regex, json, errno, warnings
from deepdiff import DeepDiff
from dictdiffer import diff
import warnings

from IPython.display import display
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pylab as plt
import matplotlib

from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import datetime
import copy
import time

from robusta import utils, metrics

import seaborn as sns
from jupyterthemes import jtplot
jtplot.style('gruvboxd')
matplotlib.use('nbagg')


__all__ = ['plot_fold', 'print_fold']





def plot_fold(cv):

    if not cv.plot:
        return

    scores = cv.get_results('score')
    k_folds = len(scores)
    n_folds = cv.n_folds

    if k_folds == 1:
        cv.fig, cv.ax = plt.subplots(1,1)

        cv.ax.set_title('Cross-Validation ({})'.format(type(cv).__name__))
        cv.ax.set_ylabel(cv.metric_name)
        cv.ax.set_xlabel('fold')
        cv.ax.grid(alpha=0.3)

    xx = np.arange(k_folds) + 1
    yy = np.abs(scores)

    #if 'n_repeats' in cv.cv_params:
    #    n_repeats = cv.cv_params['n_repeats']
    #else:
    #    n_repeats = 1

    if 'n_splits' in cv.cv_params:
        n_splits = cv.cv_params['n_splits']
    else:
        n_splits = n_folds

    k_repeats = (k_folds - 1) // n_splits
    palette = sns.color_palette()
    colors = [palette[(k_fold-1) // n_splits] for k_fold in xx]
    colors = sns.color_palette(colors) # as_cmap=True

    sns.barplot(x=xx, y=yy, palette=colors, ax=cv.ax)
    cv.fig.canvas.draw()

    if k_folds == n_folds:
        # final mean score
        score, std = cv.score(return_std=True)
        score = abs(score)

        cv.ax.axhline(score, color='w')
        cv.ax.axhline(score + std, color='w', linestyle='--', alpha=0.5)
        cv.ax.axhline(score - std, color='w', linestyle='--', alpha=0.5)

        cv.ax.axhspan(score - 1*std, score + 1*std, facecolor='w', alpha=0.1)
        cv.ax.axhspan(score - 2*std, score + 2*std, facecolor='w', alpha=0.1)
        cv.fig.canvas.draw()
        
        time.sleep(1)





def print_fold(cv):

    verbose = cv.verbose
    n_folds = cv.n_folds
    k_folds = len(cv.results)


    if verbose >= 1 and k_folds == 1:
        # model name
        print(utils.ctime_str(), cv.model_name)


    if verbose >= 2:
        # fold score
        trn, val = cv.folds[k_folds-1]
        score_abs = abs(cv.results[-1]['score'])

        time_mean = np.mean(cv.get_results('time'))
        eta = time_mean * (n_folds - k_folds)
        eta = utils.sec_to_str(eta)

        msg = (utils.ctime_str(), k_folds, n_folds, len(trn), len(val), cv.metric_name, score_abs, eta)
        print('%s fold: %i/%i    train/test: %i/%i    %s: %.4f    eta: %s' % msg)


    if verbose >= 1 and k_folds == n_folds:
        # final score
        score, score_std = cv.score(return_std=True)
        score_abs = abs(score)

        msg = (utils.ctime_str(), cv.metric_name, score_abs, score_std)
        print('%s %s: %.4f ± %.4f' % msg)
