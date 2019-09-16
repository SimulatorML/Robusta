import numpy as np
import pandas as pd

from imblearn.base import BaseSampler, check_sampling_strategy
from imblearn import under_sampling, over_sampling, combine




def make_sampler(Sampler):
    '''Wrapper for imblearn sampler, that takes and returns pandas DataFrames.

    Parameters
    ----------
    Sampler : class
        Sampler class (not instance!)

    **params :
        Set the parameters of core sampler.

    '''
    class PandasSampler(Sampler):

        def fit_resample(self, X, y):

            if isinstance(X, pd.core.frame.DataFrame):
                x_cols = X.columns
                x_dtypes = X.dtypes
            else:
                raise TypeError('X must be pandas DataFrame')

            if isinstance(y, pd.core.series.Series):
                y_type = 'series'
                y_name = y.name
            elif isinstance(y, pd.core.frame.DataFrame):
                y_type = 'frame'
                y_cols = y.columns
            else:
                raise TypeError('y must be pandas Series or DataFrame')

            X, y, _ = self._check_X_y(X, y)

            self.sampling_strategy_ = check_sampling_strategy(
                self.sampling_strategy, y, self._sampling_type)

            X_res, y_res = self._fit_resample(X, y)

            X_res = pd.DataFrame(X_res, columns=x_cols).astype(x_dtypes)

            if y_type is 'series':
                y_res = pd.Series(y_res, name=y_name)
            elif y_type is 'frame':
                y_res = pd.DataFrame(y_res, columns=y_cols)

            return X_res, y_res

    return PandasSampler



ADASYN = make_sampler(over_sampling.ADASYN)
SMOTE = make_sampler(over_sampling.SMOTE)
SVMSMOTE = make_sampler(over_sampling.SVMSMOTE)
SMOTENC = make_sampler(over_sampling.SMOTENC)
BSMOTE = make_sampler(over_sampling.BorderlineSMOTE)
ROS = make_sampler(over_sampling.RandomOverSampler)



ClusterCentroids = make_sampler(under_sampling.ClusterCentroids)
RUS = make_sampler(under_sampling.RandomUnderSampler)
IHT = make_sampler(under_sampling.InstanceHardnessThreshold)
NearMiss = make_sampler(under_sampling.NearMiss)
TomekLinks = make_sampler(under_sampling.TomekLinks)
ENN = make_sampler(under_sampling.EditedNearestNeighbours)
RENN = make_sampler(under_sampling.RepeatedEditedNearestNeighbours)
AllKNN = make_sampler(under_sampling.AllKNN)
OSS = make_sampler(under_sampling.OneSidedSelection)
CNN = make_sampler(under_sampling.CondensedNearestNeighbour)
NCR = make_sampler(under_sampling.NeighbourhoodCleaningRule)



SMOTEENN = make_sampler(combine.SMOTEENN)
SMOTETomek = make_sampler(combine.SMOTETomek)
