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
        score = '{:.{prec}f} ± {:.{prec}}'.format(trial['score'],
                                                  trial['score_std'],
                                                  prec=fs.n_digits)
        score = colored(score, 'blue') if (fs.best_iter_ is k-1) else score

        # Estimated time of arrival (ETA)
        if hasattr(fs, 'max_time') and fs.max_time:
            eta0 = max(0, (fs.max_time - fs.time_))
        else:
            eta0 = np.inf

        if hasattr(fs, 'max_iter') and fs.max_iter:
            eta1 = max(0, (fs.time_ / k) * (n - k))
        else:
            eta1 = np.inf

        eta = min(eta0, eta1)
        if eta < np.inf:
            eta = secfmt(eta)
            eta = '      eta: {}'.format(eta)
        else:
            eta = ''

        msg = 'iter: {}      features: {}      score: {}{}'
        msg = msg.format(iters, n_features, score, eta)

        logmsg(msg)

    if fs.verbose >= 10:
        # Last feature subset
        print(trial['subset'])
        print()
