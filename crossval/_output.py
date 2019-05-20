import pandas as pd
import numpy as np

import datetime, time


__all__ = ['CVLogger']




class CVLogger(object):

    def __init__(self, folds, estimator, X, X_new=None, verbose=1):
        self.n_folds = len(list(folds))
        self.n_train = len(X)
        self.n_test = len(X_new) if X_new is not None else 0

        self.est_name = _extract_est_name(estimator)
        self._log_msg(self.est_name)

        self.messages = {}
        self.last_ind = -1

        self.verbose = verbose

        self.busy = False


    def log(self, ind, result):

        if self.busy:
            raise
        else:
            self.busy = True

        sep = ' '*6

        # Fold index
        if ind > -1:
            msg = 'fold_{}'.format(ind)
        else:
            msg = 'full  '.format()

        # Train/test size
        if ind > -1:
            train, test = result['fold']
            n_train, n_test = len(train), len(test)
        else:
            n_train, n_test = self.n_train, self.n_test

        msg += '{}train/test: {}/{}'.format(sep, n_train, n_test)

        # Scores
        if ind > -1:
            for key, score in result['score'].items():
                msg += '{}{}: {:.4f}'.format(sep, key, score)

        # Save message
        self.messages[ind] = msg

        # Full train case
        if ind is -1:
            self._log_msg(ind)
            self.busy = False
            return

        # If all previous folds are ready, print them
        for i in range(self.last_ind+1, self.n_folds):
            if i in self.messages.keys():
                if i >= ind and self.verbose >= 2:
                    self._log_ind(i)
                    self.last_ind = i
            else:
                break

        self.busy = False


    def _log_ind(self, ind):
        msg = self.messages[ind]
        self._log_msg(msg)


    def _log_msg(self, msg):
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
