import pandas as pd
import numpy as np
import weightedstats as ws

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.metrics import get_scorer

from scipy.optimize import minimize

from ..preprocessing import LabelEncoder1D

import time, datetime

__all__ = ['BlendRegressor', 'BlendClassifier']




class Blend(LinearModel):
    '''
    Base Blending Estimator

    Parameters
    ----------
    mean : string or callable (default='mean')
        Select weighted average function:

            - 'mean': Arithmetic mean
            - 'hmean': Harmonic mean
            - 'gmean': Geometric mean
            - 'median': Median

        If callable, expected signature "mean_func(x, weights, **kwargs)".

    scoring : string or None (default=None)
        Objective for optimization. If not None, then calculate the optimal
        weights for the blending.

    opt_params : dict, optional (default={'method': 'SLSQP', 'options': {'maxiter': 100000}})
        Parameters for scipy <minimize> function

    verbose : int (default=100)
        Logging period. 0: No Output. 100: Print progress each 100 steps & etc.


    Attributes
    ----------
    coef_ : Series, shape (n_features, )
        Estimated weights of blending model.

    n_iters_ : int
        Number of evaluations

    scores_ : list of float
        Evaluation results

    times_ : list of float
        Iteration time (seconds)

    '''
    def __init__(self, mean='mean', scoring=None, opt_params={'method': 'SLSQP',
        'options': {'maxiter': 100000}}, verbose=100):

        self.mean = mean
        self.scoring = scoring
        self.opt_params = opt_params
        self.verbose = verbose



    def _fit(self, X, y, weights=None):

        # Save data
        self.n_features = X.shape[1]
        self.target = y.name

        # Init weights
        self.set_weights(weights)

        # Init output
        if self.verbose:
            self.n_iters_ = 0
            self.scores_ = []
            self.times_ = []
            self.time = time.time()

            self.score(X, y)
            self._print_last()

        # Optimization (if scoring is defined)
        if self.scoring is not None:

            # Define objective & initial guess
            objective = lambda w: -self.set_weights(w).score(X, y)
            w0 = self.coef_

            # Define constraints & bounds
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w)-1}]
            bounds = [(0.,1.)]*self.n_features

            # Optimize
            result = minimize(objective, w0, bounds=bounds, constraints=cons,
                **self.opt_params)

            self.set_weights(result['x'])

            if self.verbose:
                self._print_last()

        return self


    def set_weights(self, weights=None):
        weights = _check_weights(self.n_features, weights)
        self.coef_ = weights
        self.intercept_ = .0
        return self


    def score(self, X, y):
        scorer = get_scorer(self.scoring)
        score = scorer(self, X, y)
        self._save_score(score)
        return score


    def _mean(self, x):
        mean = get_mean(self.mean)
        return mean(x, weights=self.coef_)


    def _blend(self, X):
        return self._mean(X).rename(self.target)


    def _save_score(self, score):
        self.time, start_time = time.time(), self.time
        self.times_.append(self.time - start_time)
        self.scores_.append(score)
        self.n_iters_ += 1

        if self.verbose:
            if self.n_iters_ % self.verbose is 0:
                self._print_last()


    def _print_last(self):
        i = self.n_iters_
        s = self.scores_[-1]
        t = sec_to_str(sum(self.times_))

        msg = 'iters: {}      score: {:.6f}      elapsed: {}'.format(i, s, t)
        _log_msg(msg)






class BlendRegressor(Blend, RegressorMixin):

    def fit(self, X, y, weights=None):
        '''
        X : array-like, shape = (n_samples, n_features)

        y : array-like, shape = (n_samples, )

        weights : array-like, shape = [n_features] or None
            Features weights, initial guess in optimization if scoring is
            not None, otherwise use 'weights' as final coefficients. If None,
            then features are equally weighted. Must be non-negative.

        '''
        return self._fit(X, y, weights)


    def predict(self, X):
        return self._blend(X)




class BlendClassifier(Blend, ClassifierMixin):

    # WARNING: Binary only

    def fit(self, X, y, weights=None):
        '''
        X : array-like, shape = (n_samples, n_features)

        y : array-like, shape = (n_samples, )

        weights : array-like, shape = [n_features] or None
            Features weights, initial guess in optimization if scoring is
            not None, otherwise use 'weights' as final coefficients. If None,
            then features are equally weighted. Must be non-negative.

        '''
        self.le = LabelEncoder1D()
        y = self.le.fit_transform(y)
        return self._fit(X, y, weights)


    def predict_proba(self, X):
        prob = self._blend(X)
        prob = pd.concat([1-prob, prob], axis=1)
        return prob.values


    def predict(self, X):
        prob = self._blend(X)
        pred = prob.map(round).astype(int)
        pred = self.le.inverse_transform(pred)
        return pred.values




def _check_weights(n_features, weights=None):

    # Check type
    if weights is None:
        weights = np.ones(n_features)
    elif hasattr(weights, '__iter__'):
        weights = np.array(weights)
    else:
        raise TypeError('Unknown type for weights: %s' % type(weights))

    # Check size
    assert len(weights) == n_features, 'X rows and weights must have same length'

    # Check non-negativity
    assert min(weights) >= .0, 'All weights must be non-negative'

    # Normalize
    weights = np.array(weights)/sum(weights)

    return weights




MEAN_FUNCS = {
    'mean': lambda x, weights: x.dot(weights),
    'gmean': lambda x, weights: np.exp(np.log(x).dot(weights)),
    'hmean': lambda x, weights: 1/(x**-1).dot(weights),
    'median': ws.numpy_weighted_median,
}




def get_mean(mean):
    if mean in MEAN_FUNCS:
        return MEAN_FUNCS[mean]
    elif hasattr(mean, '__call__'):
        return mean
    else:
        raise ValueError(
            'Invalid value for <mean>. Allowed string ' \
            'values are {}.'.format(mean, set(MEAN_FUNCS)))



def _log_msg(msg):
    # TODO: move to the utils
    t = datetime.datetime.now().strftime("[%H:%M:%S]")
    print(t, msg)
    time.sleep(0.1)


def sec_to_str(s):
    # TODO: move to the utils
    H, r = divmod(s, 3600)
    M, S = divmod(r, 60)
    if H:
        return '{} h {} min {} sec'.format(int(H), int(M), int(S))
    elif M:
        return '{} min {} sec'.format(int(M), int(S))
    elif S >= 1:
        return '{} sec'.format(int(S))
    else:
        return '{} ms'.format(int(S*1000))
