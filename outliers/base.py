import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, OutlierMixin


__all__ = ['OutlierDetector']




class OutlierDetector(BaseEstimator, OutlierMixin):

    _estimator_type = 'outlier_detector'

    def fit_resample(self, X, y=None):

        labels = self.fit_predict(X, y)
        out = labels < 0

        return X[~out], y[~out]
