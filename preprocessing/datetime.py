import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DatetimeConverter1D(BaseEstimator, TransformerMixin):
    """
    A transformer that converts a one-dimensional array of strings into datetime objects.

    Parameters
    ----------
    params : dict
        A dictionary of parameters to pass to the pd.to_datetime function.

    Attributes
    ----------
    params : dict
        The dictionary of parameters used to create the transformer.
    """
    def __init__(self,
                 **params):
        self.params = params

    def fit(self) -> 'DatetimeConverter1D':
        """
        Fit the transformer to the input data.

        Returns
        -------
        self : object
            Returns the transformer instance.
        """
        return self

    def transform(self,
                  x: np.array) -> pd.Series:
        """
        Transform the input data into datetime objects.

        Parameters
        ----------
        x : array-like, shape (n_samples,)
            The input data.

        Returns
        -------
        x_transformed : pandas Series, shape (n_samples,)
            The transformed output data.
        """
        return pd.to_datetime(x, **self.params)


class DatetimeConverter(BaseEstimator, TransformerMixin):
    """
    Transformer that converts specified columns of a pandas DataFrame to datetime format.

    Parameters
    ----------
    copy : bool, default=True
        Whether to make a copy of the input DataFrame or modify it in place.

    **params : dict
        Additional keyword arguments to pass to pd.to_datetime().

    Returns
    -------
    DatetimeConverter
        Returns self, the transformer instance.
    """
    def __init__(self,
                 copy: bool = True,
                 **params):
        self.params = params
        self.copy = copy

    def fit(self) -> 'DatetimeConverter':
        """
        Fit transformer to the input data.

        Returns
        -------
        self : DatetimeConverter
            The transformer instance.
        """
        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data to transform.

        Returns
        -------
        pandas.DataFrame
            Transformed data with datetime columns converted to datetime format.
        """

        # Make a copy of the input data if specified.
        X = X.copy() if self.copy else X

        # Iterate over each column in the data.
        for col in X:
            # Convert the column to datetime format.
            X[col] = pd.to_datetime(X[col], **self.params)

        # Return the transformed data.
        return X


class CyclicEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes cyclic data by transforming it into sine and cosine functions.

    Parameters:
    -----------
    delta : float, default=1
        A constant added to the denominator in the transformation to avoid division by zero.

    Attributes:
    -----------
    max_ : float
        The maximum value in the input data.
    min_ : float
        The minimum value in the input data.
    """
    def __init__(self,
                 delta: int = 1):
        self.max_ = None
        self.min_ = None
        self.delta = delta

    def fit(self,
            X: pd.DataFrame) -> 'CyclicEncoder':
        """
        Computes the maximum and minimum values in the input data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to be encoded.

        Returns
        -------
        self : CyclicEncoder
            The fitted encoder.
        """
        self.min_ = X.min()
        self.max_ = X.max()

        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input data into sine and cosine functions.

        Parameters
        ----------
        X : array-like or dataframe, shape (n_samples, n_features)
            The data to be encoded.

        Returns
        -------
        encoded : dataframe, shape (n_samples, 2 * n_features)
            The encoded data, with each feature represented as a pair of sine and cosine functions.
        """

        # Normalize data to the range [0,1] and add delta to the denominator to avoid division by zero
        X = (X - self.min_) / (self.max_ - self.min_ + self.delta)

        # Compute cosine and sine functions for each feature and concatenate into a single dataframe
        return pd.concat([np.cos(X).rename(lambda x: x + '_cos', axis=1),
                          np.sin(X).rename(lambda x: x + '_sin', axis=1)],
                         axis=1).sort_index(axis=1)
