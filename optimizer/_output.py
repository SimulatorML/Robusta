from IPython import display

from matplotlib.ticker import MaxNLocator
import matplotlib.pylab as plt

import numpy as np
import time, datetime

from scipy import signal, stats

import termcolor

from robusta import utils




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
    trial = trials.iloc[-1]

    n_iters = opt.max_trials
    k_iters = len(trials)

    if opt.verbose >= 1:

        # number of performed iters
        if n_iters:
            iters = '%i/%i' % (k_iters, n_iters)
            iters_left = n_iters - k_iters
        else:
            iters = '%i' % k_iters
            iters_left = None

        # ETA (estimated time to arrival)
        iters_time = np.array(opt.trials_['time'])

        if opt.max_time:
            eta_time = max(0, opt._time_left())
        else:
            eta_time = np.nan

        if n_iters:
            iter_time = np.median(iters_time[iters_time != None])
            eta_iter = iter_time * iters_left
        else:
            eta_iter = np.nan

        eta = np.nanmin([eta_time, eta_iter])
        if not np.isnan(eta):
            eta = 'eta: {}'.format(utils.secfmt(eta))
        else:
            eta = ''

        # status & scores
        t = datetime.datetime.now().strftime("[%H:%M:%S]")

        if trial['status'] is 'ok':

            score = trial['score']
            scoring = opt.scoring if isinstance(opt.scoring, str) else 'score'

            is_best = opt.n_trials_ == (opt.best_trial_ + 1)

            s = '{:.4f}'.format(s) # TODO: custom <n_digits>
            #s = termcolor.colored(s, 'yellow') if is_best else s
            # FIXME: optuna library blocks colored output (termcolor)
            s = '[{}]'.format(s) if is_best else ' {} '.format(s)

            msg = 'iter: {}      {}: {}      {}'.format(iters, scoring, s, eta)
            utils.logmsg(msg)

        else:
            msg = 'iter: {}      status: {}'.format(iters, trial['status'])
            utils.logmsg(msg)

    if opt.verbose >= 2:
        # current hyperparams
        print('\t', trial['params'])
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
