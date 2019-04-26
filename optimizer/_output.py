from IPython import display
from matplotlib.ticker import MaxNLocator
import matplotlib.pylab as plt
import numpy as np
import time

from scipy import signal, stats

from robusta import utils, metrics




__all__ = ['print_progress', 'plot_progress']


def print_progress(opt):
    '''
    Print last trial score in optimizer.

    Parameters
    ----------
    opt : instance
        Optimizator instance.

    '''

    if len(opt.trials_) == 0:
        return

    trials = opt.trials_
    n_iters = opt.max_trials
    k_iters = len(trials)
    trial = trials[-1]

    if opt.verbose >= 1:

        # number of performed iters
        if n_iters:
            iters = '%i/%i' % (k_iters, n_iters)
            iters_left = n_iters - k_iters
        else:
            iters = '%i' % k_iters
            iters_left = None

        # eta (estimated time to arrival)
        iters_time = np.array(opt.get_key_trials('time'))

        if opt.max_time:
            eta_time = max(0, opt.max_time - opt.time_)
        else:
            eta_time = np.nan

        if n_iters:
            iter_time = np.median(iters_time[iters_time != None])
            eta_iter = iter_time * iters_left
        else:
            eta_iter = np.nan

        eta = np.nanmin([eta_time, eta_iter])
        if not np.isnan(eta):
            eta = 'eta: %s' % utils.sec_to_str(eta)
        else:
            eta = ''

        # status & scores
        if trial['status'] is 'ok':
            score = trial['score']
            best_score = opt.best_score_
            metric_name = opt.cv.metric_name
            star = '(*)' if score == best_score else '   '

            msg = (utils.ctime_str(), iters, metric_name, abs(score), abs(best_score), star, eta)
            print('%s iter: %s      %s: %.4f      best: %.4f %s   %s' % msg)
        else:
            msg = (utils.ctime_str(), iters, trial['status'])
            print('%s iter: %s      status: %s' % msg)

    if opt.verbose >= 2:
        # current hyperparameters
        print(trial['params'])
        print()



def plot_progress(opt, cut=.75, delay=10):
    '''
    Plot last trial score in optimizer

    Parameters
    ----------
    opt : instance
        Optimizator instance

    cut : float, optional
        Scores ratio to cut off. For example, 0.75 (by default) means
        that only top 25% of trials will be shown

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

        opt.ax.set_title('Hyperparameter optmization ({})'.format(type(opt).__name__))
        opt.ax.set_xlabel('# iters')
        opt.ax.set_ylabel(opt.cv.metric_name)

    # Last point
    scores = np.array(opt.scores_, dtype=np.float64)
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
        color = metrics.metric_color[opt.metric_name]
    else:
        # New ordinary trial
        line_style = '.'
        color = 'w'

    # More relevant, more brighter
    alpha = (stats.percentileofscore(scores, score)/100)**2

    opt.ax.plot(xx, np.abs(yy), line_style, c=color, alpha=alpha)
    opt.ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # force int x-axis
    opt.fig.canvas.draw()

    if iters > 10:
        # Cut plot (y-axis)
        y_min = abs(stats.scoreatpercentile(scores, 100*cut))
        y_max = abs(opt.best_score_)
        dy = abs(y_max-y_min)*0.05

        opt.ax.set_ylim(min(y_min, y_max)-dy, max(y_min, y_max)+dy)
        opt.fig.canvas.draw()

        if opt.is_finished:
            time.sleep(1)
