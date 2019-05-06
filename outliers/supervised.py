import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from ..pipeline import make_pipeline
from .base import OutlierDetector


__all__ = ['SupervisedOutlierDetector']




class SupervisedOutlierDetector(OutlierDetector):

    _estimator_type = 'outlier_detector'

    def __init__(self, estimator, scoring, thresh):
        self.estimator = estimator
        self.scoring = scoring # by sample
        self.thresh = thresh


    def fit_predict(self, X, y):

        if hasattr(self.estimator, 'fit_predict'):
            y_pred = self.estimator.fit_predict(X, y)
        else:
            y_pred = self.estimator.fit(X, y).predict(X)

        scores = self.scoring(y, y_pred)
        scores = pd.Series(scores, index=y.index)
        labels = pd.Series(None, index=y.index)

        outliers = scores < self.thresh
        labels[outliers] = -1
        labels[~outliers] = 1

        return labels
