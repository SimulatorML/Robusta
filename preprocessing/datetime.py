import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    'DatetimeConverter1D',
    'DatetimeConverter',
    'CyclicEncoder',
]




class DatetimeConverter1D(BaseEstimator, TransformerMixin):
    def __init__(self, **params):
        self.params = params

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return pd.to_datetime(x, **self.params)


class DatetimeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, copy=True, **params):
        self.params = params
        self.copy = copy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy() if self.copy else X
        for col in X:
            X[col] = pd.to_datetime(X[col], **self.params)
        return X


class CyclicEncoder(BaseEstimator, TransformerMixin):
    """Cyclic Encoder

    Convert x to the [cos(2*pi*t), sin(2*pi*t)] pair, where t is
    pre-normalized x: t = (x - min[x])/(max[x] - min[x] + delta)

    Parameters
    ----------
    delta : float
        Distance between maximum and minimum "angle"

    """
    def __init__(self, delta=1):
        self.delta = delta # max..min distance


    def fit(self, X, y=None):

        self.min_ = X.min()
        self.max_ = X.max()

        return self


    def transform(self, X):

        X = (X - self.min_)/(self.max_ - self.min_ + self.delta)

        return pd.concat([np.cos(X).rename(lambda x: x+'_cos', axis=1),
                          np.sin(X).rename(lambda x: x+'_sin', axis=1)],
                         axis=1).sort_index(axis=1)
