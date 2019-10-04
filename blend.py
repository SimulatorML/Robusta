import pandas as pd
import numpy as np

from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin, BaseEstimator
from sklearn.linear_model.base import LinearModel
from sklearn.metrics import get_scorer

from scipy.optimize import minimize

from robusta.preprocessing import RankTransformer


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

        NOT AVAILABLE: If callable, expected signature "mean_func(X, weights)"

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

        Xt = self._set_transformer(X)
        self.n_features_ = X.shape[1]
        self.set_weights(weights)

        if self.scoring:

            self.scorer = get_scorer(self.scoring)

            w0 = self.get_weights()
            objective = lambda w: -self.set_weights(w).score(Xt, y)
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w)-1}]
            bounds = [(0., 1.)] * self.n_features_
            options = {'maxiter': self.max_iter}

            result = minimize(objective, w0,
                              method='Nelder-Mead',
                              #constraints=constraints,
                              #bounds=bounds,
                              options=options)

            self.set_weights(result['x'])
            self.result_ = result

        return self


    def set_weights(self, weights):
        if weights is None:
            self.coef_ = np.ones(self.n_features_) / self.n_features_
        else:
            self.coef_ = np.array(weights) / np.sum(weights)
        return self


    def get_weights(self):
        return self.coef_


    def score(self, X, y):
        return self.scorer(self, X, y)


    @property
    def _is_fitted(self):
        return (hasattr(self, 'result_') or not self.scoring)


    @property
    def intercept_(self):
        return .0


    def _blend(self, X):

        if self._X_transform and self._is_fitted:
            X = self._X_transform(X)

        if self._y_transform:
            y = X.dot(self.coef_)
            y = self._y_transform(y)
        else:
            y = X.dot(self.coef_)

        return y.values


    def _set_transformer(self, X):

        avg_types = ['mean', 'rank', 'gmean', 'hmean']

        if self.avg_type == 'mean':
            self._X_transform = lambda X: X
            self._y_transform = lambda y: y

        elif self.avg_type == 'hmean':
            self._X_transform = lambda X: 1/X
            self._y_transform = lambda y: 1/y

        elif self.avg_type == 'gmean':
            self._X_transform = lambda X: np.log(X)
            self._y_transform = lambda y: np.exp(y)

        elif self.avg_type == 'rank':
            self._transformer = RankTransformer().fit(X)
            self._X_transform = lambda X: self._transformer.transform(X)
            self._y_transform = lambda y: y.rank(pct=True)

        else:
            raise ValueError("Invalid value '{}' for <avg_type>. Allowed "
                             "values: ".format(self.avg_type, avg_types))

        return self._X_transform(X)




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
