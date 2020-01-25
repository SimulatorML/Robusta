import pandas as pd
import numpy as np

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import get_scorer

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import get_scorer


__all__ = [
    'CaruanaRegressor',
    'CaruanaClassifier',
]




class _BaseCaruana(LinearModel):

    '''Caruana Ensemble Selection for Regression/Classification

    Paper: https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf

    Parameters
    ----------

    scoring : str
        Objective for optimization.

    iters : int (default=100)
        Number of models in ensemble.

    init_iters : int (default=10)
        Number of core models in ensemble, which selected from whole set
        of models at the beginning. Values from range 5-25 are prefered.

        Set 0 for basic algorithm.

    colsample : float or int (default=0.5)
        Number of models, sampled on each iteration. Must be from range (0, 1].

        Set 1.0 for basic algorithm.

    replace : bool (default=True)
        Whether to reuse models, already added to the ensemble (recommended).

        Set False for basic algorithm.

    random_state : int, RandomState instance, or None (default=None)
        Pseudo-random number generator to control the subsample of models.

    verbose : int (default=1)
        Verbosity level.

    n_jobs : int or None (default=None)
        The number of jobs to use for the computation.
        `None` means 1. `-1` means using all processors.

    Attributes
    ----------
    weights_ : list of int
        Number of times each model was used.

    y_avg_ : float
        Target bias

    '''

    def __init__(self, scoring, iters=100, init_iters=10, colsample=0.5,
                 replace=True, random_state=None, verbose=1, n_jobs=-1):
        self.iters = iters
        self.init_iters = init_iters
        self.scoring = scoring
        self.colsample = colsample
        self.replace = replace
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            Stacked predictions.

        y : DataFrame or Series, shape [n_samples, ] or [n_samples, n_classes]
            Target variable

        Returns
        -------
        self
        """

        # Check data
        if self._estimator_type is 'classifier':
            self.classes_ = np.unique(y)

        self.scorer = get_scorer(self.scoring)

        self.weights_ = np.zeros(X.shape[1])
        self.y_avg_ = y.mean()

        msg = "<init_iters> must be no more than <iters>"
        assert self.init_iters <= self.iters, msg

        if not self.replace:
            msg = "<iters> must be no more than X.shape[1] (if replace=True)"
            assert self.iters <= X.shape[1], msg

        # Init subset
        scores = {}
        for k in range(X.shape[1]):
            self.weights_[k] += 1
            scores[k] = self.score(X, y)
            self.weights_[k] -= 1

        scores = pd.Series(scores).sort_values(ascending=False)
        scores = scores[:self.init_iters]
        self.weights_[scores.index] += 1

        # Core Algorithm
        for i in range(self.init_iters, self.iters):

            k_range = np.arange(X.shape[1])

            if not self.replace:
                k_range = k_range[self.weights_ == 0]

            if self.colsample < 1.0:
                p = 1 + int(len(k_range) * self.colsample)
                k_range = np.random.choice(k_range, p, replace=False)

            best_score = None
            best_k = -1

            for k in k_range:
                self.weights_[k] += 1
                score = self.score(X, y)
                self.weights_[k] -= 1

                if best_k < 0 or best_score < score:
                    best_score = score
                    best_k = k

            self.weights_[best_k] += 1

        return self

    def score(self, X, y):
        return self.scorer(self, X, y)

    def _blend(self, X):
        return X.dot(self.coef_).values + self.intercept_

    @property
    def coef_(self):
        if self.weights_.any():
            return np.array(self.weights_) / np.sum(self.weights_)
        else:
            return self.weights_

    @property
    def intercept_(self):
        return 0.0 if self.coef_.any() else self.y_avg_


class CaruanaRegressor(_BaseCaruana, RegressorMixin):

    def predict(self, X):
        return self._blend(X)


class CaruanaClassifier(_BaseCaruana, ClassifierMixin):

    def predict_proba(self, X):
        y = self._blend(X)
        return np.stack([1-y, y], axis=-1)

    def predict(self, X):
        y = self.predict_proba(X)
        return np.rint(y[:, 1]).astype(int)
