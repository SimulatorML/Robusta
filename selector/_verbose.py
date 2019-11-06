import pandas as pd
import numpy as np

from robusta.utils import logmsg, secfmt

from termcolor import colored




def _print_last(fs):

    subset = fs.trials_[-1]

    if fs.verbose >= 1:

        # Iterations
        n = fs.max_iter if hasattr(fs, 'max_iter') else None
        k = subset.idx + 1

        iters = '{}/{}'.format(k, n) if n else '{}'.format(k)

        # Number of features
        n_selected = '{}/{}'.format(len(subset), fs.n_features_)

        # Score
        scores = [trial.score     for trial in fs.trials_]
        stds   = [trial.score_std for trial in fs.trials_]

        score = '{:.{prec}f}'.format(subset.score,     prec=fs.n_digits)
        std   = '{:.{prec}f}'.format(subset.score_std, prec=fs.n_digits)

        score = colored(score, 'yellow') if subset.idx == np.argmax(scores) else score
        std   = colored(std,   'cyan')   if subset.idx == np.argmin(stds)   else std

        score = '{} ± {}'.format(score, std)

        # Estimated time of arrival (ETA)
        if hasattr(fs, 'max_time') and fs.max_time:
            eta0 = max(fs.max_time - fs.total_time_, .0)
        else:
            eta0 = np.inf

        if hasattr(fs, 'max_iter') and fs.max_iter:
            eta1 = max(fs.total_time_ * (n - k) / k,  0)
        else:
            eta1 = np.inf

        eta = min(eta0, eta1)

        if eta < np.inf:
            eta = secfmt(eta)
            eta = '      ETA: {}'.format(eta)
        else:
            eta = ''

        msg = 'ITER: {}      SUBSET: {}      SCORE: {}{}'
        msg = msg.format(iters, n_selected, score, eta)

        logmsg(msg)


    if fs.verbose >= 2:

        if hasattr(subset, 'prev') and len(subset.parents) == 1:
            new, old = set(subset), set(subset.parents[0])

            diff = new - old
            if diff: logmsg('    + {}'.format(diff))

            diff = old - new
            if diff: logmsg('    – {}'.format(diff))


    if fs.verbose >= 10:
        # Last feature subset
        print(list(subset))
        print()
