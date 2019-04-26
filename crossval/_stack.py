import os, shutil, regex, json, errno, warnings
from deepdiff import DeepDiff
from dictdiffer import diff
import warnings

import pandas as pd
import numpy as np
import datetime
import copy

from robusta import utils, metrics




def plot_scores_history(cvs, ind, point='date', k_points=4, beta=None):
    # points = 'idx'/'date'/'time'
    if point in ['date', 'time']:
        dt_list = [cv.datetime for cv in cvs]
        xx = np.array([utils.dt_flt(dt) for dt in dt_list])
    elif point is 'idx':
        xx = np.array(ind)

    # scores & best scores
    yy = np.array([cv.score() for cv in cvs])
    zz = np.array([max(yy[:i+1]) for i in range(len(yy))])

    # baseline
    z0 = np.mean(zz) - beta*np.std(zz) if beta else min(yy)
    jj = (yy >= z0)
    ii = (yy == zz)

    # if loss -> abs(score)
    yy = np.abs(yy)
    zz = np.abs(zz)

    # extended XZ-space
    dx = (max(xx)-min(xx))/len(xx)
    dz = 0
    _xx = np.concatenate([[xx[0]-dx], xx[ii], [xx[-1]+dx]], axis=None)
    _zz = np.concatenate([[zz[0]-dz], zz[ii], [zz[-1]+dz]], axis=None)

    # metric
    plt.figure()
    metric_name = cvs[0].metric_name
    plt.title(cvs[0].metric_name)
    _, c = metrics.get_metric(metric_name, return_color=True)

    # best scores line
    utils.gradient_fill(_xx, _zz, fill_color=c, alpha=0.5)
    plt.plot(_xx, _zz, color=c, lw=1.5)

    # best scores points
    plt.plot(xx[ii], yy[ii], 'o', color=c)
    plt.plot(xx[ii], yy[ii], 'o', markersize=4, color='w')
    # other scores points
    plt.plot(xx, yy, '.', color='w')

    # extended XY-space
    r = 0.05
    x0, y0 = min(xx[jj]), min(yy[jj])
    x1, y1 = max(xx[jj]), max(yy[jj])
    plt.xlim(x0 - r*(x1-x0), x1 + r*(x1-x0))
    plt.ylim(y0 - r*(y1-y0), y1 + r*(y1-y0))

    # X-ticks configuration
    if point in ['date', 'time']:
        inv_fun = utils.flt_dstr if point == 'date' else utils.flt_tstr
        x_space = np.linspace(xx[0], xx[-1], k_points)
        d_space = np.vectorize(inv_fun)(x_space)
        plt.xticks(x_space, d_space, rotation=45, size=8)
    elif point is 'idx':
        nearest_in_list = lambda a, val: min(a, key=lambda x: abs(x-val))
        x_space = np.linspace(_xx[0], _xx[-1], k_points)
        x_space = np.vectorize(lambda x: nearest_in_list(xx[ii], x))(x_space)
        plt.xticks(x_space, x_space)

    plt.show()


def display_scores_table(cvs, ind, sort=False, k_top=None, fold=False):
    scores = [cv.get_results() for cv in cvs]
    names = [cv.get_name() for cv in cvs]

    df = pd.DataFrame(scores, index=names)

    fold_cols = ['fold%i' % i for i in df.columns]
    df.columns = fold_cols
    df.insert(0, 'score', df.mean(axis=1))
    df.insert(1, 'std', df.std(axis=1))
    df = df.applymap(lambda x: '%.4f' % abs(x))

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'model'}, inplace=True)
    df.index = ind

    df.insert(1, ' ', '')
    df.insert(3, '', '±')
    df.insert(5, '  ', '')

    if sort:
        df.sort_values('score', inplace=True)

    if fold == False:
        df.drop(columns=fold_cols, inplace=True)

    if k_top is None:
        k_top = 10

    display(df.head(k_top))




def stacking(cvs, ind):
    y_oofs = [cv.y_oof for cv in cvs]
    y_subs = [cv.y_sub for cv in cvs]

    y_oof = pd.concat(y_oofs, axis=1)
    y_sub = pd.concat(y_subs, axis=1)

    names = [cv.get_name() for cv in cvs]
    #y_cols = names
    y_cols = ['{}_{}'.format(idx, name) for idx, name in zip(ind, names)]
    y_oof.columns = y_cols
    y_sub.columns = y_cols

    return y_oof, y_sub



def display_corr(cvs, ind, k_top=10, method='kendall', **sns_params):
    # white: 'w', yellow: '#f9bc41', purple: 'm'
    colors = ['#161819', 'm']
    cmap = LinearSegmentedColormap.from_list("", colors)

    scores = [cv.score() for cv in cvs]
    k_top = len(scores) if k_top is None else k_top
    ii = np.array(scores).argsort()[::-1][:k_top]
    cvs = np.array(cvs)[ii]
    ind = np.array(ind)[ii]

    names = [cv.get_name() for cv in cvs]
    y_oof, y_sub = stacking(cvs, ind)
    y = pd.concat([y_oof, y_sub], axis=0)

    plt.figure()
    sns.heatmap(y.corr(method=method), xticklabels=ind, yticklabels=names,
                cmap=cmap, annot=True, fmt='.2f', annot_kws={'size': 8})
    plt.show()


def plot_tsne(cvs, ind, **tsne_params):
    names = [cv.get_name() for cv in cvs]

    y_oof, y_sub = stacking(cvs, ind)
    y = pd.concat([y_oof, y_sub], axis=0)

    XY = TSNE(**tsne_params).fit_transform(y.T)

    plt.figure()
    plt.plot(XY[:,0], XY[:,1], 'o', c='y')
    for name, xy in zip(names, XY):
        plt.annotate(name, (xy[0], xy[1]))
    plt.show()
