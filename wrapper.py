import numpy as np
import pandas as pd

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer

__all__ = ['WrappedRegressor']



class WrappedRegressor(BaseEstimator, ClassifierMixin):

    def __init__(self, regressor, method='minmax'):
        self.regressor = regressor
        self.method = method


    def fit(self, X, y):

        le = LabelBinarizer().fit(y)
        self.classes_ = le.classes_

        self.regressor_ = clone(self.regressor).fit(X, y)

        return self


    def decision_function(self, X):
        y = self.regressor_.predict(X)
        y = y - 0.5
        return y


    def predict_proba(self, X):
        y = self.decision_function(X)

        if self.method is 'minmax':
            y = y - y.min()
            y = y / y.max()

        return np.stack([1-y, y], axis=-1)


    def predict(self, X):
        y = self.decision_function(X)
        y = 1 * (y > 0)
        return y


    @property
    def coef_(self):
        return self.regressor_.coef_


    @property
    def feature_importances_(self):
        return self.regressor_.feature_importances_
