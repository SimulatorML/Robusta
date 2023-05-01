import numpy as np
import optuna
import pandas as pd
from termcolor import colored

from ..utils import logmsg, secfmt

def _print_last(opt: optuna.Trial):
    """
    Print last trial score in optimizer.

    Parameters
    ----------
    opt : instance
        Optimizator instance.
    """
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

            score = colored(score, 'yellow') if (opt.trials_['score'].idxmax() is k - 1) else score
            std = colored(std, 'cyan') if (opt.trials_['score_std'].idxmin() is k - 1) else std

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
        print(pd.Series(trial['params'], dtype='str'))
        print()
