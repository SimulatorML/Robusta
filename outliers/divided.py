import numpy as np
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing

from sklearn.base import clone, OutlierMixin, BaseEstimator
from sklearn.cluster import KMeans


__all__ = ['DividedOutlierDetector']




class DividedOutlierDetector(BaseEstimator, OutlierMixin):

    _estimator_type = 'outlier_detector'

    def __init__(self, detector, clusterer=KMeans(random_state=0),
                 verbose=0, n_jobs=-1):
        self.detector = detector
        self.clusterer = clusterer
        self.verbose = verbose
        self.n_jobs = n_jobs


    def fit_resample(self, X, y):

        self.labels_ = self.clusterer.fit_predict(F_train, y_train)
        masks = [self.labels_ == label for label in set(self.labels_)]

        jobs = (delayed(od_path)(clone(self.detector), X, y, mask)
                for mask in masks)
        paths = Parallel(backend='multiprocessing', max_nbytes='256M', pre_dispatch='all',
                         verbose=self.verbose, n_jobs=self.n_jobs)(jobs)

        xx, yy, self.detectors_ = zip(*paths)
        X_in = pd.concat(xx)
        y_in = pd.concat(yy)

        return X_in, y_in




def od_path(detector, X, y, ind):
    X, y = X[ind], y[ind]
    labels = detector.fit_predict(X)
    out = labels < 0
    return X[~out], y[~out], detector
