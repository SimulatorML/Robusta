import numpy as np
from termcolor import colored

from ..utils import logmsg, secfmt


def _print_last(fs: object):
    """
    Print information about the last feature subset in a sequential feature selection process.

    Parameters
    ----------
    fs : SequentialFeatureSelector
        The sequential feature selector object.

    Returns
    -------
    Nothing :
        None

    """
    prec = getattr(fs, "n_digits", 4)
    subset = fs.trials_[-1]

    if fs.verbose >= 1:
        # Iterations
        n = fs.max_iter if hasattr(fs, "max_iter") else None
        k = subset.idx + 1

        iters = f"{k}/{n}" if n else f"{k}"
        iters = "ITER: " + iters

        # Number of features
        sub = f"SUBSET: {len(subset)}/{fs.n_features_}"

        # Score
        scores = [trial.score for trial in fs.trials_]
        stds = [trial.score_std for trial in fs.trials_]

        score = "{:.{prec}f}".format(subset.score, prec=prec)
        std = "{:.{prec}f}".format(subset.score_std, prec=prec)

        score = colored(score, "yellow") if subset.idx == np.argmax(scores) else score
        std = colored(std, "cyan") if subset.idx == np.argmin(stds) else std

        score = f"SCORE: {score} ± {std}"

        # Estimated time of arrival (ETA)
        if hasattr(fs, "max_time") and fs.max_time:
            eta0 = max(fs.max_time - fs.total_time_, 0.0)
        else:
            eta0 = np.inf

        if hasattr(fs, "max_iter") and fs.max_iter:
            eta1 = max(fs.total_time_ * (n - k) / k, 0)
        else:
            eta1 = np.inf

        eta = min(eta0, eta1)

        if eta < np.inf:
            eta = secfmt(eta)
            eta = "ETA: {}".format(eta)
        else:
            eta = ""

        msg = f'{iters}{" " * 6}{sub}{" " * 6}{score}{" " * 6}{eta}'
        logmsg(msg)

    if fs.verbose >= 2:
        if hasattr(subset, "prev") and len(subset.parents) == 1:
            new, old = set(subset), set(subset.parents[0])

            diff = new - old
            if diff:
                logmsg(f"    + {diff}")

            diff = old - new
            if diff:
                logmsg(f"    – {diff}")

    if fs.verbose >= 10:
        # Last feature subset
        print(list(subset))
        print()
