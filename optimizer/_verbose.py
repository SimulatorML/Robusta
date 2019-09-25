from IPython import display

from matplotlib.ticker import MaxNLocator
import matplotlib.pylab as plt

import numpy as np
import pandas as pd
import time, datetime

from scipy import signal, stats

from termcolor import colored

from robusta.utils import logmsg, secfmt




__all__ = ['print_progress', 'plot_progress']


def _print_last(opt):
    '''
    Print last trial score in optimizer.

    Parameters
    ----------
    opt : instance
        Optimizator instance.

    '''
    trial = opt.trials_.iloc[-1]

    if opt.verbose >= 1:

        # Iterations
        n = opt.max_iter if hasattr(opt, 'max_iter') else None
        k = opt.n_iters_
        iters = '{}/{}'.format(k, n) if n else '{}'.format(k)

        if trial['status'] is 'ok':

            # Score
            score = '{:.{prec}f}'.format(trial['score'], prec=opt.n_digits)
            std = '{:.{prec}f}'.format(trial['score_std'], prec=opt.n_digits)

            # FIXME: colorlog & termcolor conflict...
            # https://github.com/borntyping/python-colorlog

            score = colored(score, 'yellow') if (opt.trials_['score'].idxmax() is k-1) else score
            std = colored(std, 'cyan') if (opt.trials_['score_std'].idxmin() is k-1) else std

            score = '{} ± {}'.format(score, std)

            # Estimated time of arrival (ETA)
            if hasattr(opt, 'max_time') and opt.max_time:
                eta0 = max(0, (opt.max_time - opt.total_time_))
            else:
                eta0 = np.inf

            if hasattr(opt, 'max_iter') and opt.max_iter:
                eta1 = max(0, (opt.total_time_ / k) * (n - k))
            else:
                eta1 = np.inf

            eta = min(eta0, eta1)
            if eta < np.inf:
                eta = secfmt(eta)
                eta = '      ETA: {}'.format(eta)
            else:
                eta = ''

            msg = 'ITER: {}      SCORE: {}{}'.format(iters, score, eta)
            logmsg(msg)

        else:
            msg = 'ITER: {} - {}!'.format(iters, trial['status'])
            logmsg(msg)


    if opt.verbose >= 2:
        print(pd.Series(trial['params']))
        print()



def plot_progress(opt, cut=.25, delay=10):
    '''
    Plot last trial score in optimizer

    Parameters
    ----------
    opt : instance
        Optimizator instance

    cut : float, optional
        Scores ratio to cut off. For example, 0.25 (by default) means
        that only top-75% trials will be shown

    delay : int, optional (default: 10)
        Minimum # of iterations before cutting off

    '''

    if not opt.plot:
        return

    if not len(opt.trials_):
        return

    # Init figure (if first time)
    if not hasattr(opt, 'ax'):
        opt.fig, opt.ax = plt.subplots(1,1)

        opt.ax.set_title('Hyperparameters optmization ({})'.format(type(opt).__name__))
        opt.ax.set_xlabel('# iters')

        # Metric name
        metric_name = opt.scoring if isinstance(opt.scoring, str) else 'metric'
        opt.ax.set_ylabel(metric_name)

    # Last point
    scores = np.array(opt.trials_['score'], dtype=np.float64)
    scores[scores == None] = np.nan
    iters = len(scores)
    score = scores[-1]

    xx = [iters]
    yy = [score]

    if score == opt.best_score_ and sum(~np.isnan(scores)) > 1:
        # Previous best iter
        k_iter = np.nanargmax(scores[:-1][::-1])
        k_iter = len(scores) - k_iter - 2
        k_score = scores[k_iter]

        xx.insert(0, k_iter+1)
        yy.insert(0, k_score)

        # New best trial
        line_style = 'o-'
        #color = metrics.metric_color[opt.metric_name]
        color = '#fbb339'
    else:
        # New ordinary trial
        line_style = '.'
        color = 'w'

    # More relevant, more brighter
    alpha = (stats.percentileofscore(scores, score)/100)**2

    opt.ax.plot(xx, np.abs(yy), line_style, c=color, alpha=alpha)
    opt.ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # force int x-axis
    opt.fig.canvas.draw()

    '''if iters > delay:
        # Cut plot (y-axis)
        y_min = abs(stats.scoreatpercentile(scores, 100*cut))
        y_max = abs(opt.best_score_)
        dy = abs(y_max-y_min)*0.05

        opt.ax.set_ylim(min(y_min, y_max)-dy, max(y_min, y_max)+dy)
        opt.fig.canvas.draw()'''

    if opt.is_finished:
        time.sleep(1)
