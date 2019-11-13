import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    'DatetimeConverter1D',
    'DatetimeConverter',
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
