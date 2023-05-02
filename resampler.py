import warnings
from typing import Tuple

import pandas as pd

warnings.filterwarnings('ignore', category=DeprecationWarning)

from imblearn.base import check_sampling_strategy


# TODO: make Pandas Wrapper for Sampler


class PandasSampler:
    def __init__(self):
        """
        Initialize a PandasSampler object with default values for sampling strategy and sampling type.
        """
        self._sampling_type = None
        self.sampling_strategy = None
        self.sampling_strategy_ = None

    def fit_resample(self,
                     X: pd.DataFrame,
                     y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Resample the input DataFrame X and target Series using the specified sampling strategy and sampling type.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas DataFrame of shape (n_samples, n_features) containing the input data.
        y : pd.Series
            A pandas Series of shape (n_samples,) containing the target labels.

        Returns
        -------
        tuple:
            A tuple (X_res, y_res) where X_res is a resampled DataFrame and y_res is a resampled Series or DataFrame,
            depending on the type of y.

        Raises
        ------
        TypeError:
            If X is not a pandas DataFrame or if y is not a pandas Series or DataFrame.
        """
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

        # Check and set the sampling strategy
        X, y, _ = self._check_X_y(X, y)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type)

        # Resample the data
        X_res, y_res = self._fit_resample(X, y)

        # Convert the resampled arrays back to DataFrames/Series with the original column names and dtypes
        X_res = pd.DataFrame(X_res, columns=x_cols).astype(x_dtypes)

        if y_type is 'series':
            y_res = pd.Series(y_res, name=y_name)
        elif y_type is 'frame':
            y_res = pd.DataFrame(y_res, columns=y_cols)

        return X_res, y_res


def make_sampler(Sampler):
    """
    Wrapper for imblearn sampler, that takes and returns pandas DataFrames.

    Parameters
    ----------
    Sampler : class
        Sampler class (not instance!)

    **params :
        Set the parameters of core sampler.

    """

    return PandasSampler, Sampler
