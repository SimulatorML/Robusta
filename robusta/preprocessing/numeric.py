from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
import scipy
from dask_ml.preprocessing import PolynomialFeatures
from preprocessing import Normalizer
from scipy.special import boxcox
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import QuantileTransformer, normalize, PowerTransformer
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

_NP_INT_DTYPES = ['int64', 'int32', 'int16', 'int8', 'uint32', 'uint16', 'uint8']
_PD_INT_DTYPES = ['Int64', 'Int32', 'Int16', 'Int8', 'UInt32', 'UInt16', 'UInt8']
_FLOAT_DTYPES = ['float64', 'float32', 'float16']


class DowncastTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to downcast numeric columns in a Pandas DataFrame to the smallest possible dtype that can accommodate
    the range of values in the column.

    Parameters:
    ----------
    numpy_only : bool, default=True
        Whether to use only NumPy dtypes or also include Pandas dtypes.
    errors : {'raise', 'ignore'}, default='raise'
        Action to take on unknown/unsupported columns:
        - 'raise': raise an error.
        - 'ignore': do nothing.
    copy : bool, default=True
        Whether to make a copy of the input data.

    Attributes:
    ----------
    cols : list of str
        The list of column names in the input data.
    nums : list of str
        The list of numeric column names in the input data.
    dtypes : pandas.core.series.Series
        A Series containing the dtypes of the columns in the input data.
    """
    def __init__(self,
                 numpy_only: bool = True,
                 errors: str = 'raise',
                 copy: bool = True):
        self.numpy_only = numpy_only
        self.errors = errors
        self.copy = copy

    def fit(self,
            X: pd.DataFrame) -> 'DowncastTransformer':
        """
        Fit the transformer to the input data.

        Parameters:
        ----------
        X : pandas.core.frame.DataFrame
            The input data.

        Returns:
        ----------
        self : object
            The transformer instance.
        """

        # Store the list of column names and numeric column names in the input data X
        self.cols = list(X.columns)
        self.nums = list(X.select_dtypes(include=['int', 'float']))

        # Store the dtypes of the columns in the input data X
        self.dtypes = X.dtypes.copy()

        # Check if the value of errors parameter is valid
        errors_vals = ['raise', 'ignore']
        if self.errors not in errors_vals:
            raise ValueError('<errors> must be in {}'.format(errors_vals))

        # If there are non-numeric columns in the input data X and the value of errors parameter
        # is 'raise', raise an error
        if len(self.nums) < len(self.cols) and self.errors is 'raise':
            cols_diff = list(set(self.cols) - set(self.nums))
            raise ValueError("Found non-numeric columns {}".format(cols_diff))

        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data.

        Parameters:
        ----------
        X : pandas.core.frame.DataFrame
            The input data.

        Returns:
        ----------
        X_transformed : pandas.core.frame.DataFrame
            The transformed data.
        """

        # Check if there are new columns in the input data X that were not present in the input data used for fitting
        cols_diff = set(self.cols) ^ set(X.columns)
        nums_diff = set(self.nums) ^ set(X.select_dtypes(np.number))
        if len(cols_diff) > 0:
            raise ValueError("Found new columns {}".format(cols_diff))

        # Check if there are new numeric columns in the input data X that were not present in the input data used for
        # fitting
        if len(nums_diff) > 0:
            raise ValueError("Found new numeric columns {}".format(nums_diff))

        # For each numeric column in the input data X, determine the smallest datatype that can represent the column
        for col, x in X[self.nums].items():
            col_type = self._fit_downcast(x)
            self.dtypes[col] = col_type

        # Convert the input data X to the smallest datatype that can accommodate the range of values in each column
        return X.astype(self.dtypes, errors=self.errors, copy=self.copy)

    def _fit_downcast(self,
                      x: pd.DataFrame) -> str:
        """
        Find the smallest datatype that can represent the input array without losing precision.

        Parameters:
        -----------
        x : pd.DataFrame
            Input array to determine smallest datatype

        Returns:
        --------
        col_type : dtype
            The smallest datatype that can represent the input array without losing precision
        """

        # Determine the minimum and maximum values in the column
        x_min = x.min()
        x_max = x.max()

        # Try to convert the column to an integer type
        try:
            # Use only NumPy dtypes or also include Pandas dtypes based on the value of numpy_only parameter
            INT_DTYPES = _NP_INT_DTYPES if self.numpy_only else _PD_INT_DTYPES

            # If casting x to the first integer type in INT_DTYPES produces any differences, raise an exception
            if (x.astype(INT_DTYPES[0]) != x).any():
                raise ValueError()

            # Set the column type to the first integer type in INT_DTYPES
            col_type = INT_DTYPES[0]

            # Get the number of bits in col_type
            col_bits = np.iinfo(col_type).bits

            # Loop over integer types in INT_DTYPES
            for int_type in INT_DTYPES:
                # Get information on the integer type
                int_info = np.iinfo(int_type)

                # If the minimum value of x is greater than or equal to the minimum value of the integer type,
                # the maximum value of x is less than or equal to the maximum value of the integer type, and the
                # number of bits in the integer type is less than or equal to the number of bits in col_type,
                # set col_type to the integer type
                if (x_min >= int_info.min) \
                        and (x_max <= int_info.max) \
                        and (col_bits >= int_info.bits):
                    col_bits = int_info.bits
                    col_type = int_type

        except (Exception,):

            # Set the column type to the first float type in _FLOAT_DTYPES
            col_type = _FLOAT_DTYPES[0]

            # Get the number of bits in col_type
            col_bits = np.finfo(col_type).bits

            # Loop over float types in _FLOAT_DTYPES
            for float_type in _FLOAT_DTYPES:
                # Get information on the float type
                float_info = np.finfo(float_type)

                # If the minimum value of x is greater than or equal to the minimum value of the float type,
                # the maximum value of x is less than or equal to the maximum value of the float type, and the
                # number of bits in the float type is less than the number of bits in col_type, set col_type to
                # the float type
                if (x_min >= float_info.min) \
                        and (x_max <= float_info.max) \
                        and (col_bits > float_info.bits):
                    col_bits = float_info.bits
                    col_type = float_type

        # Return the smallest datatype that can represent the input array without losing precision
        return col_type


class QuantileTransformer(QuantileTransformer):
    """
    A class for quantile-based feature transformation.

    This class inherits from sklearn's QuantileTransformer, adding additional
    functionality to handle pandas dataframes as inputs and outputs.

    Args:
        QuantileTransformer: The base QuantileTransformer class from scikit-learn.

    Returns:
        QuantileTransformer: An instance of the QuantileTransformer class.
    """

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data X using the fitted transformer.

        Args:
            X (array-like): Input data to transform.

        Returns:
            array-like or DataFrame: The transformed input data X, in the form of an array or
            pandas DataFrame, depending on the format of the input data.
        """

        # Check if X is a pandas dataframe and set a flag to return a pandas dataframe later
        return_df = hasattr(X, 'columns')

        # If X is a pandas dataframe, store the column names and index to be used when returning a pandas dataframe later
        if return_df:
            columns = X.columns
            index = X.index

        # Check if X is a pandas dataframe and convert it to a numpy array if it is
        X = self._check_inputs(X)

        # Perform the transformation
        self._transform(X, inverse=False)

        # If X was a pandas dataframe, convert the transformed numpy array back to a pandas dataframe using the
        # stored column names and index
        if return_df:
            return pd.DataFrame(X, columns=columns, index=index)

        # If X was not a pandas dataframe, return the transformed numpy array
        else:
            return X


class GaussRankTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps each feature to a Gaussian distribution using rank-based transforms.

    Parameters
    ----------
    ranker : object, optional (default=sklearn.preprocessing.QuantileTransformer())
        The transformer used to rank the features. Must have a `fit_transform` method.
    copy : bool, optional (default=True)
        Whether to make a copy of the input data.
    eps : float, optional (default=1e-9)
        A small number to add to the denominator of the transform to avoid division by zero.

    Attributes
    ----------
    ranker_ : object
        The fitted transformer.
    """
    def __init__(self,
                 ranker: QuantileTransformer = QuantileTransformer(),
                 copy: bool = True,
                 eps: float = 1e-9):
        self.ranker = ranker
        self.copy = copy
        self.eps = eps

    def fit(self) -> 'GaussRankTransformer':
        """
        Fit the transformer to the data.

        This method does nothing and is only included for compatibility with scikit-learn.

        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using Gaussian rank-based transforms.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.

        Returns
        -------
        X_transformed : pd.DataFrame
            The transformed data.
        """

        # Copy the input data to avoid modifying the original
        if self.copy:
            X, self.ranker_ = X.copy(), clone(self.ranker)
        else:
            X, self.ranker_ = X, self.ranker

        # Rank transform the input data
        X = self.ranker_.fit_transform(X)

        # Shift and scale the transformed data to map it to a Gaussian distribution
        X -= 0.5
        X *= 2.0 - self.eps
        X = scipy.special.erfinv(X)

        # Return the transformed data
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Winsorize the input data by setting extreme values to a specified percentile value.

    Parameters
    ----------
    q_min : float, default=0.05
        The percentile value to use as the minimum threshold.
    q_max : float, default=0.95
        The percentile value to use as the maximum threshold.
    apply_test : bool, default=True
        Whether to apply the winsorizing transformation to the test data.

    Attributes
    ----------
    min_ : pandas Series
        The minimum threshold values for each column in the input data.
    max_ : pandas Series
        The maximum threshold values for each column in the input data.
    """

    def __init__(self,
                 q_min: float = 0.05,
                 q_max: float = 0.95,
                 apply_test: bool = True):
        self.apply_test = apply_test
        self.q_min = q_min
        self.q_max = q_max

    def fit_transform(self,
                      X: pd.DataFrame,
                      y: np.array = None) -> pd.DataFrame:
        """
        Fit to the input data, then winsorize the input data.

        Parameters
        ----------
        X : pandas DataFrame
            The input data to be transformed.
        y : None
            Ignored.

        Returns
        -------
        X_transformed : pandas DataFrame
            The transformed data.

        """

        # Fit to the input data
        self._fit(X)

        # Return the fitted Winsorizer transformer
        return self.transform(X)

    def fit(self,
            X: pd.DataFrame) -> 'Winsorizer':
        """
        Compute the minimum and maximum threshold values for each column in the input data.

        Parameters
        ----------
        X : pandas DataFrame
            The input data to fit to.

        Returns
        -------
        self : Winsorizer
            The fitted Winsorizer transformer.

        """

        # Fit to the input data
        self._fit(X)

        # Return the fitted Winsorizer transformer
        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorize the input data by setting values outside the specified percentile range to the corresponding
        threshold value.

        Parameters
        ----------
        X : pandas DataFrame
            The input data to be transformed.

        Returns
        -------
        X_transformed : pandas DataFrame
            The transformed data.

        """

        # If apply_test is True, apply the transformation to the input data
        if self.apply_test:
            return X.clip(self.min_, self.max_, axis=1)
        else:
            return X

    def _fit(self,
             X: pd.DataFrame) -> None:
        """
        Compute the minimum and maximum threshold values for each column in the input data.

        Parameters
        ----------
        X : pandas DataFrame
            The input data to fit to.

        Raises
        ------
        AssertionError
            If any of the parameter values are invalid.

        """

        # Ensure the parameter values are valid
        assert isinstance(self.apply_test, bool), '<apply_test> must be boolean'
        assert isinstance(self.q_min, float), '<q_min> must be float'
        assert isinstance(self.q_max, float), '<q_max> must be float'
        assert self.q_min < self.q_max, '<q_min> must be smaller than <q_max>'
        assert 0 <= self.q_min <= 1, '<q_min> must be in [0..1]'
        assert 0 <= self.q_max <= 1, '<q_max> must be in [0..1]'

        # Compute the minimum and maximum threshold values for each column in the input data
        self.min_ = X.quantile(self.q_min)
        self.max_ = X.quantile(self.q_max)


class SyntheticFeatures(BaseEstimator, TransformerMixin):
    """
    Class to generate synthetic features from input data.

    Parameters
    ----------
    pair_sum : bool, default=True
        Whether to generate features by summing pairs of columns.
    pair_dif : bool, default=True
        Whether to generate features by taking the difference of pairs of columns.
    pair_mul : bool, default=True
        Whether to generate features by multiplying pairs of columns.
    pair_div : bool, default=True
        Whether to generate features by dividing pairs of columns.
    join_X : bool, default=True
        Whether to join the synthetic features with the original data.
    eps : float, default=1e-2
        Small number added to the denominator to avoid division by zero.
    """
    def __init__(self,
                 pair_sum: bool = True,
                 pair_dif: bool = True,
                 pair_mul: bool = True,
                 pair_div: bool = True,
                 join_X: bool = True,
                 eps: float = 1e-2):
        self.pair_sum = pair_sum
        self.pair_dif = pair_dif
        self.pair_mul = pair_mul
        self.pair_div = pair_div
        self.join_X = join_X
        self.eps = eps

    def fit(self,
            X: pd.DataFrame) -> 'SyntheticFeatures':
        """
        Fit the synthetic features generator to the input data.

        Parameters
        ----------
        X : array-like or pandas DataFrame, shape (n_samples, n_features)
            Input data to fit the generator on.
        y : array-like, optional
            Targets for supervised learning, unused.

        Returns
        -------
        self : object
            Returns self to allow method chaining.
        """

        # Sets the column names for the input data
        if isinstance(X, pd.core.frame.DataFrame):
            self.columns = X.columns
        else:
            self.columns = ['x_{}'.format(i) for i in range(X.shape[1])]
        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic features from the input data.

        Parameters
        ----------
        X : array-like or pandas DataFrame, shape (n_samples, n_features)
            Input data to generate synthetic features for.

        Returns
        -------
        Xt : pandas DataFrame, shape (n_samples, n_synthetic_features + n_features)
            DataFrame containing the original data with synthetic features added.
        """

        # Sets the index of the output dataframe to match the index of the input data
        if isinstance(X, pd.core.frame.DataFrame):
            inds = X.index
        else:
            inds = np.arange(X.shape[0])
            X = pd.DataFrame(X, columns=self.columns, index=inds)

        Xt = pd.DataFrame(index=inds)

        # Generate synthetic features by summing pairs of columns
        cols_pairs = np.array(list(combinations(self.columns, 2)))
        cols_A = cols_pairs[:, 0]
        cols_B = cols_pairs[:, 1]

        # Generate synthetic features by taking the difference of pairs of columns
        if self.pair_sum:
            # generate names for new synthetic columns
            cols = ['{}+{}'.format(a, b) for a, b in cols_pairs]

            # compute pairwise sum of columns
            F = np.vstack([X[a].values + X[b].values for a, b in cols_pairs]).T

            # create dataframe with synthetic columns
            F = pd.DataFrame(F, index=inds, columns=cols)

            # join the synthetic columns with the output dataframe
            Xt = Xt.join(F)

        # if difference of columns is enabled
        if self.pair_dif:
            # generate names for new synthetic columns
            cols = ['{}-{}'.format(a, b) for a, b in cols_pairs]

            # compute pairwise difference of columns
            F = np.vstack([X[a].values - X[b].values for a, b in cols_pairs]).T

            # create dataframe with synthetic columns
            F = pd.DataFrame(F, index=inds, columns=cols)

            # join the synthetic columns with the output dataframe
            Xt = Xt.join(F)

        # if multiplication of columns is enabled
        if self.pair_mul:
            # generate names for new synthetic columns
            cols = ['{}*{}'.format(a, b) for a, b in cols_pairs]

            # compute pairwise multiplication of columns
            F = np.vstack([X[a].values * X[b].values for a, b in cols_pairs]).T

            # create dataframe with synthetic columns
            F = pd.DataFrame(F, index=inds, columns=cols)

            # join the synthetic columns with the output dataframe
            Xt = Xt.join(F)

        # if division of columns is enabled
        if self.pair_div:
            # generate names for new synthetic columns
            cols = ['{}/{}'.format(a, b) for a, b in cols_pairs]

            # compute pairwise division of columns, adding self.eps to the denominator to avoid division by zero
            F = np.vstack([X[a].values / (X[b].values + self.eps) for a, b in cols_pairs]).T

            # create dataframe with synthetic columns
            F = pd.DataFrame(F, index=inds, columns=cols)

            # join the synthetic columns with the output dataframe
            Xt = Xt.join(F)

            # generate additional names for new synthetic columns (inverse division)
            cols = ['{}/{}'.format(a, b) for b, a in cols_pairs]

            # compute pairwise division of columns (inverse), adding self.eps to the denominator to avoid division by zero
            F = np.vstack([X[a].values / (X[b].values + self.eps) for b, a in cols_pairs]).T

            # create dataframe with synthetic columns
            F = pd.DataFrame(F, index=inds, columns=cols)

            # join the synthetic columns with the output dataframe
            Xt = Xt.join(F)

        # if joining of synthetic columns with original data is enabled
        if self.join_X:
            # join the synthetic columns with the original data
            Xt = X.join(Xt)

        # return the dataframe with synthetic columns added
        return Xt


class RobustScaler(BaseEstimator, TransformerMixin):
    """
    Scale features using statistics that are robusta to outliers.

    This implementation is similar to `sklearn.preprocessing.RobustScaler`, but with a few differences:
    - It allows for custom quantiles to be used for determining the range of the data to be scaled.
    - It includes an `eps` parameter that is used to avoid division by zero in case the range is very small.
    - It can handle both pandas DataFrames and numpy arrays.

    Parameters
    ----------
    centering : bool, default=True
        Whether to center the data before scaling.
    scaling : bool, default=True
        Whether to scale the data to the specified range.
    quantiles : tuple of floats, default=(0.25, 0.75)
        The quantiles used to determine the range of the data to be scaled.
    copy : bool, default=True
        Whether to copy the input data before transforming it.
    eps : float, default=1e-3
        A small number to add to the denominator when scaling, to avoid division by zero.

    Attributes
    ----------
    center_ : pandas Series or numpy array of shape (n_features,)
        The median value of each feature in the training set.
    scale_ : pandas Series or numpy array of shape (n_features,)
        The range of each feature in the training set, as determined by the specified quantiles.
    """

    def __init__(self,
                 centering: bool = True,
                 scaling: bool = True,
                 quantiles: tuple = (0.25, 0.75),
                 copy: bool = True,
                 eps: float = 1e-3):
        self.centering = centering
        self.scaling = scaling
        self.quantiles = quantiles
        self.copy = copy
        self.eps = eps

    def fit(self,
            X: pd.DataFrame) -> 'RobustScaler':
        """
        Compute the median and range of each feature in the training set.

        Parameters
        ----------
        X : pandas DataFrame or numpy array of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        self : RobustScaler
            The fitted scaler.
        """

        # Extracts the minimum and maximum quantiles and checks if they are valid
        q_min, q_max = self.quantiles
        if not 0 <= q_min <= q_max <= 1:
            raise ValueError(f"Invalid quantiles: {self.quantiles}")

        # Computes the median of each feature if centering is enabled
        if self.centering:
            self.center_ = X.median()

        # Computes the range of each feature if scaling is enabled
        if self.scaling:
            self.scale_ = X.quantile(q_max) - X.quantile(q_min)
            # Replaces any value below a certain threshold with 1 to avoid division by zero
            self.scale_[self.scale_ < self.eps] = 1

        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the input data using the median and range computed in the `fit` method.

        Parameters
        ----------
        X : pandas DataFrame or numpy array of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_scaled : pandas DataFrame or numpy array of shape (n_samples, n_features)
            The scaled input data.
        """

        # Creates a copy of the input data if necessary
        X = X.copy() if self.copy else X

        # If centering is enabled, compute the median value of each feature.
        if self.centering:
            X -= self.center_

        # If scaling is enabled, compute the range of each feature as the difference between the maximum and minimum
        # values, as determined by the specified quantiles. If the range is too small, add a small epsilon value to
        # avoid division by zero.
        if self.scaling:
            X /= self.scale_

        return X


class StandardScaler(BaseEstimator, TransformerMixin):
    """
    A class for scaling the features of a dataset to have zero mean and unit variance.

    Parameters:
    ----------
    with_mean : bool, default=True
        If True, center the data by subtracting the mean.
    with_std : bool, default=True
        If True, scale the data to have unit variance.
    copy : bool, default=True
        If True, make a copy of the input data.

    Attributes:
    ----------
    mean_ : array, shape (n_features,)
        The mean of each feature.
    std_ : array, shape (n_features,)
        The standard deviation of each feature.
    """

    def __init__(self,
                 with_mean: bool = True,
                 with_std: bool = True,
                 copy: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def fit(self,
            X: pd.DataFrame) -> 'StandardScaler':
        """
        Compute the mean and standard deviation of each feature from the input data.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        -------
        self : object
            Returns the instance itself.
        """

        # Compute the mean of each feature
        self.mean_ = X.mean() if self.with_mean else None

        # Compute the standard deviation of each feature
        self.std_ = X.std() if self.with_std else None

        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the input data to have zero mean and unit variance.

        Parameters:
        ----------
        X : pd.DataFrame
            The input data.

        Returns:
        -------
        X_scaled : pd.DataFrame
            The scaled input data.
        """

        # Make a copy of the input data, if specified
        X = X.copy() if self.copy else X

        # Center the data by subtracting the mean, if specified
        if self.with_mean:
            X -= self.mean_

        # Scale the data to have unit variance, if specified
        if self.with_std:
            X /= self.std_

        return X


class MinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Scales input data to a range between 0 and 1 using minimum and maximum values.

    Parameters:
    -----------
    copy : bool, optional (default=True)
        If True, makes a copy of input data before transforming it.

    Attributes:
    -----------
    min_ : float or ndarray of shape (n_features,)
        Minimum value of each feature in the training set.
    max_ : float or ndarray of shape (n_features,)
        Maximum value of each feature in the training set.
    std_ : float or ndarray of shape (n_features,)
        Range (maximum - minimum) of each feature in the training set.
    """

    def __init__(self,
                 copy: bool = True):
        self.copy = copy

    def fit(self,
            X: pd.DataFrame) -> 'MinMaxScaler':
        """
        Compute the minimum, maximum, and range of each feature in the input data.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data to be scaled.

        Returns:
        --------
        self : MinMaxScaler
            The fitted MinMaxScaler object.
        """

        # Compute the minimum, maximum, and range of each feature in the input data.
        self.min_ = X.min()
        self.max_ = X.max()
        self.std_ = self.max_ - self.min_

        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data to a scaled range between 0 and 1.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data to be transformed.

        Returns:
        --------
        X_scaled : array-like of shape (n_samples, n_features)
            Scaled input data.
        """

        # Make a copy of the input data if the copy parameter is set to True.
        X = X.copy() if self.copy else X

        # Scale the input data to a range between 0 and 1 using the computed minimum and range.
        X -= self.min_
        X /= self.std_

        return X


class MaxAbsScaler(BaseEstimator, TransformerMixin):
    """
    Scale features by dividing each feature by the maximum absolute value of that feature in the input data.

    Parameters:
    -----------
    copy : bool, default=True
        Whether to create a copy of the input data.

    Attributes:
    ----------
    scale_ : pandas Series
        Per-feature maximum absolute values used for scaling.

    """

    def __init__(self,
                 copy: bool = True):
        self.copy = copy

    def fit(self,
            X: pd.DataFrame) -> 'MaxAbsScaler':
        """
        Compute the maximum absolute values of each feature in the input data.

        Parameters:
        -----------
        X : pd.DataFrame
            The input data.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """

        # Compute the minimum and maximum values for each feature in X
        a = X.min()
        b = X.max()

        # Concatenate a and b along the columns axis and compute the maximum absolute value for each row
        self.scale_ = pd.concat([a, b], axis=1).abs().max(axis=1)

        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the input data.

        Parameters:
        -----------
        X : pd.DataFrame
            The input data.

        Returns:
        --------
        X : array-like or sparse matrix, shape [n_samples, n_features]
            The scaled input data.
        """

        # Create a copy of the input data if copy is True
        X = X.copy() if self.copy else X

        # Divide each feature by its maximum absolute value (stored in self.scale_)
        X /= self.scale_

        return X


class Normalizer(Normalizer):
    """
    A class that applies normalization to the rows of a matrix or DataFrame.

    Parameters
    ----------
    copy : bool, default=True
        Whether to create a copy of the input array or DataFrame or perform normalization in-place.
    norm : {'l1', 'l2', 'max'}, default='l2'
        The type of normalization to apply. 'l1' and 'l2' norms are standard normalization methods, while 'max'
        scales each row by the maximum absolute value in that row.
    """

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization to the rows of a matrix or DataFrame.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            The input data to normalize.

        Returns
        -------
        X_normalized : array-like or DataFrame, shape (n_samples, n_features)
            The normalized input data.
        """

        # Check if the input data is a DataFrame and keep track of the columns and index
        return_df = hasattr(X, 'columns')

        if return_df:
            columns = X.columns
            index = X.index

        # Apply normalization to the input data using the specified parameters
        X = check_array(X, accept_sparse='csr')
        X = normalize(X, axis=1, copy=self.copy, norm=self.norm)

        # If the input data is a DataFrame, convert the output back to a DataFrame with original columns and index
        if return_df:
            return pd.DataFrame(X, columns=columns, index=index)
        else:
            return X


class KBinsDiscretizer1D(BaseEstimator, TransformerMixin):
    """
    Discretizes a 1D array-like input into 'bins' number of equal frequency or equal width intervals.

    Parameters:
    -----------
    bins : int (default=5)
        The number of bins to divide the input into.
    strategy : str (default='quantile')
        The method used to determine the bins.
        - 'quantile': Each bin has approximately the same number of samples.
        - 'uniform': Each bin has the same width.

    Attributes:
    -----------
    bins_ : ndarray, shape (n_bins + 1,)
        The computed bins.
    """
    def __init__(self,
                 bins: int = 5,
                 strategy: str = 'quantile'):
        self.bins = bins
        self.strategy = strategy

    def fit(self,
            y: pd.Series) -> 'KBinsDiscretizer1D':
        """
        Compute the bins based on the input data y.

        Parameters:
        -----------
        y : pd.Series
            The input data to be discretized.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """

        # check if the strategy is 'quantile' to compute bins with equal number of samples in each bin
        if self.strategy is 'quantile':
            _, self.bins_ = pd.qcut(y, self.bins, retbins=True, duplicates='drop')

        # check if the strategy is 'uniform' to compute bins with equal width
        elif self.strategy is 'uniform':
            _, self.bins_ = pd.cut(y, self.bins, retbins=True, duplicates='drop')

        # if the strategy is neither 'quantile' nor 'uniform', raise an error
        else:
            raise ValueError(f"Unknown strategy value passed: {self.strategy}")

        return self

    def transform(self,
                  y: pd.Series) -> np.array:
        """
        Transform the input data y using the computed bins.

        Parameters:
        -----------
        y : array-like, shape (n_samples,)
            The input data to be discretized.

        Returns:
        --------
        binned_y : array-like, shape (n_samples,)
            The discretized input data.
        """

        # discretize y using the previously computed bins
        return pd.cut(y, self.bins_)


class KBinsDiscretizer(KBinsDiscretizer1D):
    """
    A class that discretizes continuous features into bins using the KBinsDiscretizer1D class.

    Parameters:
    -----------
    All parameters are inherited from the KBinsDiscretizer1D class.
    """

    def __init__(self):
        super().__init__()

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series = None) -> 'KBinsDiscretizer':
        """
        Fit the discretizer on a pandas DataFrame.

        Parameters:
        -----------
        X : pandas DataFrame, shape (n_samples, n_features)
            The input data to be discretized.

        y : None
            Ignored variable.

        Returns:
        --------
        self : object
            Returns self.
        """

        # Initialize transformers and bins_ attributes
        self.transformers = {}
        self.bins_ = {}

        # Loop over columns in X
        for col in X:
            # Create a KBinsDiscretizer1D object and fit on the column data
            params = self.get_params()
            transformer = KBinsDiscretizer1D(**params).fit(X[col])

            # Store the fitted transformer and its computed bins in the transformers and bins_ attributes
            self.transformers[col] = transformer
            self.bins_[col] = transformer.bins_

        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a pandas DataFrame using the fitted discretizer.

        Parameters:
        -----------
        X : pandas DataFrame, shape (n_samples, n_features)
            The input data to be transformed.

        Returns:
        --------
        Xt : pandas DataFrame, shape (n_samples, n_features)
            The transformed data.
        """

        # Initialize a new DataFrame with the same index as X
        Xt = pd.DataFrame(index=X.index)

        # Loop over columns in X
        for col, transformer in self.transformers.items():
            # Transform the column using the corresponding transformer and store in the new DataFrame
            Xt[col] = transformer.transform(X[col])
        return Xt


class PowerTransformer(PowerTransformer):
    """
    Subclass of sklearn.preprocessing.PowerTransformer that adds support for returning transformed data as a DataFrame.

    Parameters
    ----------
    method : str {'box-cox', 'yeo-johnson'}, default='yeo-johnson'
        The method used to transform the data.
    standardize : bool, default=True
        If True, the transformed data is standardized.
    copy : bool, default=True
        If False, the input data is overwritten.
    """
    def fit_transform(self,
                      X: pd.DataFrame,
                      y: Optional[pd.Series] = None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : pd.DataFrame
            The data to transform.
        y : pd.Series, optional (default=None)
            Target data

        Returns
        -------
        transformed : pandas.DataFrame of shape (n_samples, n_features) or numpy.ndarray of shape (n_samples, n_features)
            The transformed data. If the input data is a pandas DataFrame, the output will be a DataFrame with the same
            columns and index.
        """

        # Check if the input is a pandas DataFrame and store the column names and index values
        return_df = hasattr(X, 'columns')
        if return_df:
            columns = X.columns
            index = X.index

        # Fit the transformer and transform the input data
        X = self._fit(X, y, force_transform=True)

        # If the input is a pandas DataFrame, return the transformed data as a DataFrame with the same columns and index
        if return_df:
            return pd.DataFrame(X, columns=columns, index=index)

        # If the input is not a pandas DataFrame, return the transformed data as a numpy array
        else:
            return X

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to transform.

        Returns
        -------
        transformed : pandas.DataFrame of shape (n_samples, n_features) or numpy.ndarray of shape (n_samples, n_features)
            The transformed data. If the input data is a pandas DataFrame, the output will be a DataFrame with the same
            columns and index.
        """

        # Check if the input is a pandas DataFrame and store the column names and index values
        return_df = hasattr(X, 'columns')
        if return_df:
            columns = X.columns
            index = X.index

        # Check if the transformer is fitted and transform the input data
        check_is_fitted(self, 'lambdas_')
        X = self._check_input(X, check_positive=True, check_shape=True)

        # Select the transformation function based on the specified method
        transform_function = {'box-cox': boxcox,
                              'yeo-johnson': self._yeo_johnson_transform
                              }[self.method]

        # Apply the selected transformation function to each feature using the corresponding lambda value
        for i, lmbda in enumerate(self.lambdas_):
            with np.errstate(invalid='ignore'):  # hide NaN warnings
                X[:, i] = transform_function(X[:, i], lmbda)

        # If standardize is True, standardize the transformed data
        if self.standardize:
            X = self._scaler.transform(X)

        # If the input is a pandas DataFrame, return the transformed data as a DataFrame with the same columns
        if return_df:
            return pd.DataFrame(X, columns=columns, index=index)
        else:
            return X

        return X


class Binarizer(BaseEstimator, TransformerMixin):
    """
    A class for binarizing data by thresholding.

    Parameters:
    -----------
    threshold : float, optional (default=0.0)
        The threshold value for binarization.
    """

    def __init__(self,
                 threshold: float = 0.0):
        self.threshold = threshold

    def fit(self) -> 'Binarizer':
        """
        Fits the Binarizer transformer to the input data.

        Returns:
        --------
        self : object
            Returns self.
        """
        return self

    def transform(self,
                  X: pd.DataFrame) -> np.ndarray:
        """
        Binarizes the input data based on the threshold value.

        Parameters:
        -----------
        X : array-like or sparse matrix, shape [n_samples, n_features]
            The input data.

        Returns:
        --------
        binarized_data : ndarray, shape [n_samples, n_features]
            The binarized data as an array of unsigned integers.
        """
        return (X > self.threshold).astype('uint8')


class PolynomialFeatures(PolynomialFeatures):
    """
    Generate polynomial and interaction features.

    Parameters:
    -----------
    degree : int, optional (default=2)
        The degree of the polynomial features.

    interaction_only : bool, optional (default=False)
        If True, only interaction features are produced.

    include_bias : bool, optional (default=True)
        If True (default), then include a bias column, i.e. a column of ones.

    preserve_dataframe : bool, optional (default=True)
        If True (default), then the output is returned as a pandas DataFrame
        with column names derived from the input feature names. If False, then
        the output is returned as a NumPy array.

    """
    def __init__(self,
                 degree: int = 2,
                 interaction_only: bool = False,
                 include_bias: bool = True,
                 preserve_dataframe: bool = True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.preserve_dataframe = preserve_dataframe
