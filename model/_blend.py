import pandas as pd
import numpy as np
import weightedstats as ws

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.metrics import get_scorer

from scipy.optimize import minimize

from ..preprocessing import LabelEncoder1D


__all__ = ['BlendRegressor', 'BlendClassifier']




class Blend(LinearModel):
    '''
    Base Blending Estimator

    Parameters
    ----------
    scoring : string or None (default=None)
        Objective for optimization. If not None, then calculate the optimal
        weights for the blending.

    Attributes
    ----------
    coef_ : Series, shape (n_features, )
        Estimated weights of blending model.

    '''
    def __init__(self, scoring=None, **opt_params):

        self.scoring = scoring
        self.opt_params = opt_params



    def _fit(self, X, y, weights=None):
        '''
        X : array-like, shape = (n_samples, n_features)

        y : array-like, shape = (n_samples, )

        weights : array-like, shape = [n_features] or None
            Features weights, initial guess in optimization if scoring is
            not None, otherwise use 'weights' as final coefficients. If None,
            then features are equally weighted. Must be non negative.

        '''

        # Save data
        self.n_cols = X.shape[1]
        self.target = y.name

        # Initial weights
        weights = np.ones(self.n_cols) if weights is None else weights
        weights = norm_weights(weights)
        self._set_weights(weights)

        # Optimization (if scoring is defined)
        if self.scoring is not None:

            # Define scorer & objective
            scorer = get_scorer(self.scoring)
            objective = lambda w: -scorer(self._set_weights(w), X, y)

            # Define constraints & bounds
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w)-1}]
            bounds = [(0.,1.)]*self.n_cols

            # Define optimizator params
            opt_params = {'method': 'SLSQP', 'options': {'maxiter': 100000}}
            opt_params.update(self.opt_params)

            # Optimize
            result = minimize(objective, self.coef_, constraints=constraints,
                bounds=bounds, **opt_params)

            self._set_weights(result['x'])

        return self


    def _set_weights(self, weights=None):
        self.coef_ = weights
        self.intercept_ = .0
        return self


    def _mean(self, x):
        return np.average(x, weights=self.coef_)


    def _blend(self, X):
        return X.apply(self._mean, axis=1).rename(self.target)




class BlendRegressor(Blend, RegressorMixin):

    def fit(self, X, y, weights=None):
        return self._fit(X, y, weights)

    def predict(self, X):
        return self._blend(X)




class BlendClassifier(Blend, ClassifierMixin):

    # WARNING: Binary only

    def fit(self, X, y, weights=None):
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




#def _check_weights(x, weights=None):
#    w = ones_weights(x) if weights is None else weights
#    w = norm_weights(w)

# Check type
#if weights is None:
#    weights = ones_weights(self.X_shape_[1])
#elif hasattr(weights, '__iter__'):
#    weights = np.array(weights)
#else:
#    raise TypeError('Unknown type for weights: %s' % type(w))

# Check size
#msg = 'X and weights have incompatible shapes: \
#    {} and {}'.format(self.X_shape_, np.shape(weights))
#assert len(weights) == self.X_shape_[1],

# Check non-negativity
#assert min(weights) >= .0, 'All weights must be non-negative'

ones_weights = lambda x: np.ones(len(x))
norm_weights = lambda weights: np.array(weights)/sum(weights)

#def _check_weights(x, weights=None):
#    w = ones_weights(x) if weights is None else weights
#    w = norm_weights(w)
#    return w

def weighted_gmean(x, weights=None):
    w = _check_weights(x, weights)
    return np.exp(sum(w*np.log(x)))

#def weighted_hmean(x, weights=None):
#    w = _check_weights(x, weights)
#    return (sum(w*x**-1))**-1

#MEAN_FUNCS = {
#    'mean': np.average,
#    'gmean': weighted_gmean,
#    'hmean': weighted_hmean,
#    'median': ws.numpy_weighted_median,
#}
