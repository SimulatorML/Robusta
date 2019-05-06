import numpy as np
import pandas as pd

from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from .base import OutlierDetector


__all__ = [
    'RobustCovariance',
    'LocalOutlierFactor',
    'IsolationForest',
    'OneClassSVM',
]


class OneClassSVM(OneClassSVM, OutlierDetector): pass
class RobustCovariance(EllipticEnvelope, OutlierDetector): pass
class LocalOutlierFactor(LocalOutlierFactor, OutlierDetector): pass
class IsolationForest(IsolationForest, OutlierDetector): pass
