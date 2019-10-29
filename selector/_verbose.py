import pandas as pd
import numpy as np

from robusta.utils import logmsg, secfmt

from termcolor import colored




def _print_last(fs):

    trial = fs.trials_.iloc[-1]

    if fs.verbose >= 1:

        # Iterations
        n = fs.max_iter if hasattr(fs, 'max_iter') else None
        k = fs.n_iters_
        iters = '{}/{}'.format(k, n) if n else '{}'.format(k)

        # Number of features
        n_features = '{}/{}'.format(len(trial['subset']), fs.n_features_)

        # Score
        score = '{:.{prec}f}'.format(trial['score'], prec=fs.n_digits)
        std = '{:.{prec}f}'.format(trial['score_std'], prec=fs.n_digits)

        score = colored(score, 'yellow') if (fs.trials_['score'].idxmax() is k-1) else score
        std = colored(std, 'cyan') if (fs.trials_['score_std'].idxmin() is k-1) else std

        score = '{} ± {}'.format(score, std)

        # Estimated time of arrival (ETA)
        if hasattr(fs, 'max_time') and fs.max_time:
            eta0 = max(0, (fs.max_time - fs.total_time_))
        else:
            eta0 = np.inf

        if hasattr(fs, 'max_iter') and fs.max_iter:
            eta1 = max(0, (fs.total_time_ / k) * (n - k))
        else:
            eta1 = np.inf

        eta = min(eta0, eta1)
        if eta < np.inf:
            eta = secfmt(eta)
            eta = '      ETA: {}'.format(eta)
        else:
            eta = ''

        msg = 'ITER: {}      SUBSET: {}      SCORE: {}{}'
        msg = msg.format(iters, n_features, score, eta)

        logmsg(msg)


    if fs.verbose >= 2 and getattr(trial, 'prev_subset', None) is not None:

        new, old = trial['subset'], trial['prev_subset']

        diff = new - old
        if diff: logmsg('ADD : {}'.format(diff))

        diff = old - new
        if diff: logmsg('DROP: {}'.format(diff))


    if fs.verbose >= 10:
        # Last feature subset
        print(trial['subset'])
        print()
