import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from imblearn.base import check_sampling_strategy


# TODO: make Pandas Wrapper for Sampler


class PandasSampler():

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



def make_sampler(Sampler):
    '''Wrapper for imblearn sampler, that takes and returns pandas DataFrames.

    Parameters
    ----------
    Sampler : class
        Sampler class (not instance!)

    **params :
        Set the parameters of core sampler.

    '''

    return PandasSampler, Sampler
