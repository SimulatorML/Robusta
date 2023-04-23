from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from .base import OutlierDetector


# class OneClassSVM(OneClassSVM, OutlierDetector): pass
# class RobustCovariance(EllipticEnvelope, OutlierDetector): pass
# class LocalOutlierFactor(LocalOutlierFactor, OutlierDetector): pass
# class IsolationForest(IsolationForest, OutlierDetector): pass
