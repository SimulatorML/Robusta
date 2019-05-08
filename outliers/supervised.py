import numpy as np
import pandas as pd

from sklearn.base import clone, BaseEstimator

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

        self.estimator_ = clone(self.estimator).fit(X, y)
        y_pred = self.estimator_.predict(X)

        scores = self.scoring(y, y_pred)
        scores = pd.Series(scores, index=y.index)
        labels = pd.Series(None, index=y.index)

        outliers = scores < self.thresh
        labels[outliers] = -1
        labels[~outliers] = 1

        return labels
