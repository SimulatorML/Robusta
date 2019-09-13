import pandas as pd
import numpy as np

from robusta.utils import logmsg, secfmt

from termcolor import colored




def _print_last(fs):

    trial = fs.trials_.iloc[-1]

    if fs.verbose >= 1:

        # Iterations
        n = fs.max_iter
        k = fs.n_iters_
        iters = '{}/{}'.format(k, n) if n else '{}'.format(k)

        # Number of features
        n_features = '{}/{}'.format(len(trial['subset']), fs.n_features_)

        # Score
        score = trial['score']
        score = '{:.4f}'.format(score)
        score = colored(score, 'green') if (fs.best_iter_ is k-1) else score

        # Estimated time of arrival (ETA)
        if fs.max_time or fs.max_iter:
            eta0 = max(0, (fs.time_ / k) * (n - k) if fs.max_iter else np.inf)
            eta1 = max(0, (fs.max_time - fs.time_) if fs.max_time else np.inf)
            eta = min(eta0, eta1)
            eta = secfmt(eta)
            eta = '      eta: {}'.format(eta)
        else:
            eta = ''

        msg = 'iter: {}      features: {}      score: {}{}'
        msg = msg.format(iters, n_features, score, eta)

        logmsg(msg)

    if fs.verbose >= 2:
        # Last feature subset
        print(trial['subset'])
        print()
