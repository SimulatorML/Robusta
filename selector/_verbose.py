import pandas as pd
import numpy as np

from robusta.utils import logmsg, secfmt




def _print_last(fs):

    trial = fs.trials_.iloc[-1]

    if fs.verbose >= 1:

        iters = '{}'.format(fs.n_trials_)
        iters = iters + '/{}'.format(fs.max_trials) if fs.max_trials else ''

        k = len(trial['subset'])
        n = fs.n_features_

        score = trial['score']

        if fs.max_time or fs.max_trials:
            eta0 = (fs.time_ / k) * (n - k) if fs.max_trials else np.inf
            eta1 = (fs.max_time - fs.time_) if fs.max_time else np.inf
            eta = min(eta0, eta1)
            eta = secfmt(eta)
            eta = '    eta: {}'.format(eta)
        else:
            eta = ''

        msg = 'iter: {}    features: {}/{}    score: {:.4f}{}'
        msg = msg.format(iters, k, n, score, eta)

        logmsg(msg)

    if fs.verbose >= 2:
        # Last feature subset
        print(trial['subset'])
        print()
