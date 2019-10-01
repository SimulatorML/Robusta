import pandas as pd
import numpy as np

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.metrics import get_scorer

from scipy.optimize import minimize


__all__ = ['BlendingRegressor', 'BlendingClassifier']




class BaseBlending(LinearModel):
    '''Base Blending Estimator

    Parameters
    ----------
    mean : string or callable (default='mean')
        Select weighted average function:

            - 'mean': Arithmetic mean
            - 'rank': Ranking mean
            - 'hmean': Harmonic mean
            - 'gmean': Geometric mean
            - 'median': Median

        If callable, expected signature "mean_func(x, weights, **kwargs)"

    scoring : string or None (default=None)
        Objective for optimization. If None, all weights are equal.
        Otherwise, calculate the optimal weights for blending.

    verbose : int (default=100)
        Logging period. 0 means no output.
        100: print progress each 100 iters.

    opt_params : dict, optional (default={'method': 'SLSQP', 'options': {'maxiter': 100000}})
        Parameters for scipy <minimize> function


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
    def __init__(self, avg_type='mean', max_iter=10000, scoring=None, verbose=100):

        self.avg_type = avg_type
        self.max_iter = max_iter
        self.scoring = scoring
        self.verbose = verbose


    def fit(self, X, y, weights=None):

        if self._estimator_type is 'classifier':
            self.classes_ = np.unique(y)

        self.avg = check_avg(self.avg_type)

        self.n_features_ = X.shape[1]
        self.weights = weights

        if self.scoring:

            self.scorer = get_scorer(self.scoring)

            def objective(w):
                self.weights = w
                score = self.score(X, y)
                return -score

            w0 = np.array(self.coef_)

            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w)-1}]
            bounds = [(0., 1.)] * self.n_features_
            options = {'maxiter': self.max_iter}

            result = minimize(objective, w0,
                              method='Nelder-Mead',
                              options=options,
                              #constraints=cons,
                              #bounds=bounds,
                             )

            #score = -objective(result['x'])

        return self


    @property
    def intercept_(self):
        return .0


    @property
    def coef_(self):
        if self.weights is None:
            weights = np.ones(self.n_features_)
        else:
            weights = np.array(self.weights)
        return weights / np.sum(weights)


    def score(self, X, y):
        return self.scorer(self, X, y)


    def _blend(self, X):
        return self.avg(X, self.coef_).values






class BlendingRegressor(BaseBlending, RegressorMixin):

    def predict(self, X):
        return self._blend(X)




class BlendingClassifier(BaseBlending, ClassifierMixin):

    def predict_proba(self, X):
        y = self._blend(X)
        return np.stack([1-y, y], axis=-1)

    def predict(self, X):
        y = self.predict_proba(X)
        return np.rint(y[:, 1]).astype(int)




AVG_TYPES = {
    'mean': lambda X, w: X.dot(w),
    'rank': lambda X, w: X.rank(pct=True).dot(w).rank(pct=True),
    'gmean': lambda X, w: np.exp(np.log(X).dot(w)),
    'hmean': lambda X, w: 1/(X**-1).dot(w),
}




def check_avg(avg_type):
    if avg_type in AVG_TYPES:
        return AVG_TYPES[avg_type]
    elif hasattr(avg_type, '__call__'):
        return avg_type
    else:
        raise ValueError('Invalid value for <avg_type>. '
                         'Allowed values: {}.'.format(list(AVG_TYPES)))
