import pandas as pd
import numpy as np

from sklearn.base import (
    clone,
    is_regressor,
    is_classifier,
    BaseEstimator,
    TransformerMixin,
    RegressorMixin,
    ClassifierMixin,
)



class StackingTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, estimators, cv=None, scoring=None):
        pass


class StackingRegressor(StackingTransformer, RegressorMixin):

    def __init__(self, estimators, cv=None, scoring=None):
        pass


class StackingClassifier(StackingTransformer, ClassifierMixin):

    def __init__(self, estimators, cv=None, scoring=None):
        pass
