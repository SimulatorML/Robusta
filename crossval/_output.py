import pandas as pd
import numpy as np

import datetime, time


__all__ = ['CVLogger']




class CVLogger(object):

    def __init__(self, folds, verbose=1):
        self.n_folds = len(list(folds))
        self.verbose = verbose

        self.messages = {}
        self.last_ind = -1

        self.busy = False


    def log(self, ind, result):

        if ind is -1:
            return

        while True:
            if not self.busy:
                self.busy = True
                break
            else:
                time.sleep(0.1)

        sep = ' '*6

        # Fold index
        msg = 'fold_{}'.format(ind)

        # Train/test size
        train, test = result['fold']
        n_train, n_test = len(train), len(test)

        msg += '{}train/test: {}/{}'.format(sep, n_train, n_test)

        # Scores
        for key, score in result['score'].items():
            msg += '{}{}: {:.4f}'.format(sep, key, score)

        # Save message
        self.messages[ind] = msg

        # If all previous folds are ready, print them
        for i in range(self.last_ind+1, self.n_folds):
            if i in self.messages.keys():
                if self.verbose >= 2 and i >= ind:
                    self._log_ind(i)
                    self.last_ind = i
            else:
                break

        self.busy = False


    def _log_ind(self, ind):
        msg = self.messages[ind]
        _log_msg(msg)



def _log_msg(msg):
    t = datetime.datetime.now().strftime("[%H:%M:%S]")
    print(t, msg)
    time.sleep(0.1)



def _extract_est_name(estimator, drop_type=False):
    """Extract name of estimator instance.

    Parameters
    ----------
    estimator : estimator object
        Estimator or Pipeline

    drop_type : bool (default=False)
        Whether to remove an ending of the estimator's name, contains
        estimator's type. For example, 'XGBRegressor' transformed to 'XGB'.


    Returns
    -------
    name : string
        Name of the estimator

    """
    name = estimator.__class__.__name__

    if name is 'Pipeline':
        last_step = estimator.steps[-1][1]
        name = _extract_est_name(last_step, drop_type=drop_type)

    elif name is 'TransformedTargetRegressor':
        regressor = estimator.regressor
        name = _extract_est_name(regressor, drop_type=drop_type)

    elif drop_type:
        for etype in ['Regressor', 'Classifier', 'Ranker']:
            if name.endswith(etype):
                name = name[:-len(etype)]

    return name
