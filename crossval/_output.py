import pandas as pd
import numpy as np

import datetime, time
import termcolor

from ..model import extract_model_name, extract_model


__all__ = ['CVLogger']




class CVLogger(object):

    def __init__(self, folds, verbose=1, prec=6):
        self.n_folds = len(list(folds))
        self.verbose = verbose
        self.prec = prec

        self.messages = {}
        self.last_ind = -1

        self.busy = False


    def log(self, ind, result, n_digits=6):

        if ind is -1:
            return

        while True:
            if not self.busy:
                self.busy = True
                break
            else:
                time.sleep(0.1)

        # Fold index & score(s)
        msg = 'FOLD {}:'.format(ind)
        msg = _agg_scores(msg, result['score'], self.prec)

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


    def log_start(self, estimator, scorer):

        if self.verbose >= 2:
            est_name = extract_model_name(estimator, short=False)
            _log_msg(est_name)
            print()


    def log_final(self, results):

        scores = results['score']

        means = {metric: scores.mean() for metric, scores in scores.items()}
        stds = {metric: scores.std() for metric, scores in scores.items()}

        if self.verbose >= 2:
            print()

            # Mean Score
            msg = _agg_scores('MEAN  :', means, self.prec, colored=True)
            _log_msg(msg)

            # Std Score
            msg = _agg_scores('STD   :', stds, self.prec)
            _log_msg(msg)

            print()

        elif self.verbose == 1:

            for metric in scores:

                mean, std = means[metric], stds[metric]
                m = '{:.{prec}f}'.format(mean, prec=self.prec)
                s = '{:.{prec}f}'.format(std, prec=self.prec)

                msg = '{} ± {}'.format(m, s) + ' ({})'.format(metric)
                _log_msg(msg)


    def _log_ind(self, ind):
        msg = self.messages[ind]
        _log_msg(msg)



def _log_msg(msg):
    for m in msg.split('\n'):
        t = datetime.datetime.now().strftime("[%H:%M:%S]")
        print(t, m)
        time.sleep(0.01)


def _agg_scores(msg, scores, prec, colored=False):

    for metric, score in scores.items():

        scr = '{:.{prec}f}'.format(score, prec=prec)
        scr = termcolor.colored(scr, 'yellow') if colored else scr

        msg += ' '*(3 - 1*(score < 0))
        msg += scr

        msg += ' ({})'.format(metric)

    return msg



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
