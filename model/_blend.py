#from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearModel

import weightedstats as ws
import numpy as np
import pandas as pd

from scipy.optimize import minimize

#from robusta.metrics import get_metric


__all__ = ['Blend']




ones_weights = lambda x: np.ones(len(x))
norm_weights = lambda weights: np.array(weights)/sum(weights)

def weighted_gmean(x, weights=None):
    w = ones_weights(x) if weights is None else weights
    w = norm_weights(w)
    return np.exp(sum(w*np.log(x)))


def weighted_hmean(x, weights=None):
    w = ones_weights(x) if weights is None else weights
    w = norm_weights(w)
    return (sum(w*x**-1))**-1

def weighted_rms(x, weights=None):
    w = ones_weights(x) if weights is None else weights
    w = norm_weights(w)
    return np.sqrt(sum(w*x**2))


mean_funcs = {
    'mean': np.average,
    'median': ws.numpy_weighted_median,
    'gmean': weighted_gmean,
    'hmean': weighted_hmean,
    'rms': weighted_rms,
}



class Blend(LinearModel):
    '''
    Blending Estimator

    Parameters
    ----------
    mean_func : string, optional
        The weighted mean function:

        'mean': [default]
            x = sum[x_i * w_i] / sum[w_i]
        'median':
            x = x_k, where sum[w_i] == 1 and k satisfying:
            sum[w_i, i=1..k-1] <= 0.5
            sum[w_i, i=k+1..n] <= 0.5
        'gmean':
            x = exp( sum[w_i * log(x_i)] / sum[w_i] )
        'hmean':
            x = sum[w_i] / sum[w_i / x_i]
        'rms':
            sqrt( sum[w_i * x_i^2] )

    eval_metric : string or None
        Objective for optimization. If not None, then calculate the optimal
        weights for the blending.

    max_iters : int, optional (defautl: 1000)
        Maximum number of iterations in optimization. This parameter is ignored
        when eval_metric is None.

    Attributes
    ----------
    coef_ : Series, shape (n_features, )
        Estimated weights of blending model.

    '''
    def __init__(self, mean_func='mean', eval_metric=None, max_iters=500):

        self.eval_metric = eval_metric
        self.mean_func = mean_func
        self.max_iters = max_iters



    def fit(self, X, y, weights=None):
        '''
        X : array-like, shape = (n_samples, n_features)

        y : array-like, shape = (n_samples, )

        weights : array-like, shape = [n_features] or None
            Features weights, initial guess in optimization if eval_metric is
            not None, otherwise use 'weights' as final coefficients. If None,
            then features are equally weighted. Must be non negative.

        '''

        X_shape = np.shape(X)
        n_features = X_shape[1]

        # Check type
        if isinstance(weights, type(None)):
            weights = np.ones(n_features)
        elif isinstance(weights, dict):
            weights = list(weights.values())
        elif hasattr(weights, '__iter__'):
            weights = list(weights)
        else:
            raise TypeError('Unknown type for weights: %s' % type(w))

        # Check size
        w_shape = np.shape(weights)
        msg = 'X and weights have incompatible shapes: {} and {}'.format(X_shape, w_shape)
        assert n_features == w_shape[0], msg

        # Check non-negativity
        assert min(weights) >= .0, 'All weights must be non-negative'

        # Blending function
        self._mean = lambda x, weights: mean_funcs[self.mean_func](x, weights=weights)
        self._blend = lambda X, weights: X.apply(lambda x: self._mean(x, weights), axis=1)

        # Optimization (if eval_metric is defined)
        if self.eval_metric:
            metric = get_metric(self.eval_metric)
            objective = lambda weights: metric(y, self._blend(X, weights))
            result = minimize(objective, weights, bounds=[(0.,1.)]*n_features,
                options={'maxiter': self.max_iters})
            weights = result['x']

        # Define final linear model
        self.coef_ = norm_weights(weights)
        self.intercept_ = .0

        return self
