from typing import Union, List, Iterable, Callable, Mapping, Sequence, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that selects specific columns from a Pandas DataFrame.

    Parameters
    ----------
    columns : str or list, default=None
        The name(s) of the column(s) to select. If None, returns the entire DataFrame.

    Attributes
    ----------
    columns : list
        The name(s) of the column(s) to select.

    Raises
    ------
    KeyError
        If any of the specified columns are not present in the input DataFrame.
    """
    def __init__(self,
                 columns: Union[list, str] = None) -> None:
        self.columns = columns

    def fit(self) -> 'ColumnSelector':
        """Return instance"""
        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by selecting the specified columns.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.

        Returns
        -------
        X_selected : pd.DataFrame
            The transformed DataFrame with only the selected columns.

        Raises
        ------
        KeyError
            If any of the specified columns are not present in the input DataFrame.
        """

        # If no columns are specified, return the entire DataFrame
        if self.columns is None:
            return X

        # If only one column is specified as a string, convert it to a list
        elif isinstance(self.columns, str):
            columns = [self.columns]

        # Otherwise, use the list of column names
        else:
            columns = list(self.columns)

        # Check if all specified columns are present in the input DataFrame
        missing_cols = set(columns) - set(X.columns)
        if missing_cols:
            # Raise a KeyError with a message indicating the missing columns
            raise KeyError(f"The DataFrame does not include the columns: {missing_cols}")

        # Return a new DataFrame with only the selected columns
        return X[columns]


class TypeSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that selects columns of a specified data type from a Pandas DataFrame.

    Parameters:
    dtype (type or list of types): The data type(s) of the column(s) to select.

    Attributes:
        dtypes (list of types): The data type(s) to select.
        columns_ (pandas Index): The selected columns.

    Raises:
        ValueError: If dtype is not a type or a list of types.
    """
    def __init__(self,
                 dtype: Union[type, List[type]]):
        self.dtypes = None
        self.columns_ = None
        self.dtype = dtype

    def fit(self,
            X: pd.DataFrame) -> 'TypeSelector':
        """
        Fit method for the TypeSelector transformer.

        Selects columns of the specified data type(s) from the input DataFrame.

        Parameters:
            X (pandas DataFrame): The input data to fit the transformer.

        Returns:
            self (TypeSelector): The fitted transformer object.
        """

        # Set the dtypes attribute to the specified data type(s) as a list.
        self.dtypes = self.dtype if isinstance(self.dtype, list) else [self.dtype]

        # Select columns of the specified data type(s) and store the result in the columns_ attribute.
        self.columns_ = X.select_dtypes(include=self.dtypes).columns
        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method for the TypeSelector transformer.

        Selects columns of the specified data type(s) from the input DataFrame.

        Parameters:
            X (pandas DataFrame): The input data to transform.

        Returns:
            X_selected (pandas DataFrame): The selected columns of the input data.
        """

        # Select the columns of the specified data type(s) from the input DataFrame.
        X_selected = X[self.columns_]
        return X_selected


class TypeConverter(BaseEstimator, TransformerMixin):
    """
    Transformer that converts the data types of a pandas DataFrame.

    Parameters:
    -----------
    dtypes : dict
        A dictionary mapping column names to desired data types.

    Attributes:
    -----------
    dtypes_old_ : pandas Series
        A pandas Series containing the original data types of the DataFrame.
    """
    def __init__(self,
                 dtypes: dict):
        """
        Initialize the TypeConverter transformer.

        Parameters:
        -----------
        dtypes : dict
            A dictionary mapping column names to desired data types.
        """
        self.dtypes_old_ = None
        self.dtypes = dtypes

    def fit(self,
            X: pd.DataFrame) -> 'TypeConverter':
        """
        Fit the transformer to a pandas DataFrame.

        Parameters:
        -----------
        X : pandas DataFrame
            The DataFrame to fit the transformer to.

        Returns:
        --------
        self : TypeConverter
            The fitted transformer.
        """
        self.dtypes_old_ = X.dtypes
        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformer to a pandas DataFrame.

        Parameters:
        -----------
        X : pandas DataFrame
            The DataFrame to apply the transformer to.

        Returns:
        --------
        X_transformed : pandas DataFrame
            The transformed DataFrame.
        """
        return X.astype(self.dtypes)

    def inverse_transform(self,
                          X: pd.DataFrame) -> pd.DataFrame:
        """
        Revert the transformer on a pandas DataFrame to the original data types.

        Parameters:
        -----------
        X : pandas DataFrame
            The DataFrame to revert the transformer on.

        Returns:
        --------
        X_original : pandas DataFrame
            The DataFrame with the original data types.
        """
        return X.astype(self.dtypes_old_)


class Identity(BaseEstimator, TransformerMixin):
    """
   Transformer that returns the input array as is.

   This transformer is useful as a placeholder or identity function in a
   pipeline. It simply returns the input array without modification.
   """
    def fit(self) -> 'Identity':
        """
        This method does nothing and returns the transformer object.

        Returns
        -------
        self : object
            Returns the transformer object.
        """
        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the input array X without modification.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples.

        Returns
        -------
        X : pd.DataFrame
            The unchanged input samples.
        """
        return X


class ColumnRenamer(BaseEstimator, TransformerMixin):
    """
    A transformer that renames the columns of a pandas DataFrame.

    Parameters
    ----------
    column : callable, list, str, dict, optional
        The column names to use for the output DataFrame. If None, the input
        DataFrame column names will be used.
        - If callable, should take a column name as input and return a new name.
        - If list, should be a sequence of column names of the same length as
          the input DataFrame.
        - If str, should be a prefix to use for the column names. The column
          names will be the prefix followed by a number.
        - If dict, should be a dictionary mapping input column names to output
          column names.
        Default is None.
    prefix : str, optional
        A prefix to add to the new column names, before the column name.
        Default is ''.
    suffix : str, optional
        A suffix to add to the new column names, after the column name.
        Default is ''.
    copy : bool, optional
        Whether to make a copy of the input DataFrame before modifying it.
        Default is True.

    Attributes
    ----------
    mapper_ : dict
        A dictionary that maps input column names to output column names.
    """
    def __init__(
            self,
            column: Optional[Sequence[str]] = None,
            prefix: str = '',
            suffix: str = '',
            copy: bool = True,
    ) -> None:
        self.column = column
        self.prefix = prefix
        self.suffix = suffix
        self.copy = copy

    def fit(self,
            X: pd.DataFrame) -> 'ColumnRenamer':
        """
        Fit the transformer to the input DataFrame.

        Parameters
        ----------
        X : pandas DataFrame
            The input DataFrame to fit the transformer to.

        Returns
        -------
        self : ColumnRenamer
            The fitted transformer instance.

        Raises
        ------
        ValueError
            If an unknown <column> type is passed.

        """

        # Define a dictionary that maps input column names to output column names
        if isinstance(self.column, Callable):
            feature_names = [self.column(x) for x in X]
        elif isinstance(self.column, (list, tuple)):
            feature_names = list(map(str, self.column))
        elif isinstance(self.column, str):
            feature_names = [f"{self.column}{i}" for i, _ in enumerate(X.columns)]
        elif isinstance(self.column, Mapping):
            feature_names = [self.column.get(f, f) for f in X.columns]
        else:
            raise ValueError(f'Unknown <column> type passed: {type(self.column)}')

        # Create a mapper dictionary to map old names to new names
        self.mapper_ = {old_name: self.prefix + new_name + self.suffix for old_name, new_name in
                        zip(X.columns, feature_names)}
        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by renaming its columns.

        Parameters
        ----------
        X : pandas DataFrame
            The input DataFrame to transform.

        Returns
        -------
        X_new : pandas DataFrame
            The transformed DataFrame with renamed columns.

        """

        # Make a copy of the input DataFrame if copy is True
        X = X.copy() if self.copy else X

        # Use the mapper dictionary to rename the columns of the input DataFrame
        X.columns = X.columns.map(self.mapper_)
        return X


class ColumnFilter(BaseEstimator, TransformerMixin):
    """
    Transformer that filters the columns of a pandas DataFrame based on a given function.

    Parameters
    ----------
    func : callable
       Function that takes a column name as input and returns a boolean indicating whether
       the column should be kept or not.
    **kwargs
       Additional keyword arguments to pass to the function.

    Attributes
    ----------
    features : list
       List of column names to keep, determined by applying the `func` function to the
       input DataFrame's columns.
    """
    def __init__(self,
                 func: Callable,
                 **kwargs):
        self.func = func
        self.kwargs = kwargs

    def fit(self,
            X: pd.DataFrame) -> 'ColumnFilter':
        """
        Compute the list of columns to keep based on the input DataFrame `X`, and return
        the transformer object.

        Parameters
        ----------
        X : pandas DataFrame
            Input DataFrame from which to extract the column names.

        Returns
        -------
        self : ColumnFilter
            The transformer object.
        """

        # Filter the input DataFrame columns based on the provided function and store the result in `self.features`
        self.features = list(filter(self.func, X.columns))
        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a new DataFrame with only the columns in `self.features`.

        Parameters
        ----------
        X : pandas DataFrame
            Input DataFrame from which to extract the columns.

        Returns
        -------
        X_transformed : pandas DataFrame
            Transformed DataFrame with only the columns in `self.features`.
        """

        # Select only the columns in `self.features` from the input DataFrame and return the result
        return X[self.features]


class SimpleImputer(BaseEstimator, TransformerMixin):
    """
    SimpleImputer is a transformer that replaces missing values in a dataset with a specified
    value or statistic. The transformer operates on a pandas dataframe.
    """
    def __init__(self,
                 strategy: str = 'mean',
                 fill_value: Optional[float] = None,
                 copy: bool = True):
        self.fill_value_ = None
        self.inplace = None
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy

    def fit(self,
            X: pd.DataFrame) -> 'SimpleImputer':
        """
        Learns the fill value for the transformer based on the input dataset.

        Parameters:
        X (pandas.DataFrame): The dataset to learn from.

        Returns:
        self: Returns an instance of self.
        """

        # set inplace to the opposite of copy
        self.inplace = not self.copy

        # if the strategy is mean or median, calculate the fill value as the mean or median of each column
        if self.strategy in ['mean', 'median']:

            # check that all columns have numeric values, otherwise raise a ValueError
            if X.isna().any().any():
                raise ValueError("With strategy '{}' all columns must "
                                 "have numeric values.".format(self.strategy))

            else:
                # calculate the fill value based on the selected strategy
                self.fill_value_ = X.mean() if self.strategy == 'mean' else X.median()

        # if the strategy is mode, calculate the fill value as the most common value in each column
        elif self.strategy == 'mode':
            self.fill_value_ = X.mode().iloc[0]

        # if the strategy is const, use the specified fill value
        elif self.strategy == 'const':
            self.fill_value_ = self.fill_value

        # if the strategy is unknown, raise a ValueError
        else:
            raise ValueError("Unknown strategy '{}'".format(self.strategy))

        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the learned fill value to the input dataset.

        Parameters:
        X (pandas.DataFrame): The dataset to transform.

        Returns:
        pandas.DataFrame: The transformed dataset.
        """

        # fill missing values with the learned fill value, using inplace if specified and downcasting to integer if possible
        return X.fillna(self.fill_value_, inplace=self.inplace, downcast='infer')


class ColumnGrouper(BaseEstimator, TransformerMixin):
    """
    A transformer that groups DataFrame columns into a MultiIndex based on the given group name.

    Parameters
    ----------
    group : str or iterable of str
        The name of the group to assign to each column, or a list of group names in the order of the columns.
    copy : bool, default=True
        Whether to copy the input DataFrame before transforming.

    Attributes
    ----------
    features_ : pd.MultiIndex
        The MultiIndex representing the grouped columns.
    """
    def __init__(self,
                 group: Union[str, Iterable],
                 copy: bool = True):
        self.group = group
        self.copy = copy
        self.features_ = None

        if isinstance(self.group, str):
            self.groups_ = [self.group]
        elif isinstance(self.group, Iterable):
            self.groups_ = list(self.group)

    def fit(self,
            X: pd.DataFrame) -> 'ColumnGrouper':
        """
        Compute the MultiIndex based on the columns of the input DataFrame X.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to be transformed.

        Returns
        -------
        self : ColumnGrouper
            The fitted transformer instance.
        """

        # Assign the MultiIndex representing the grouped columns to the features_ attribute.
        self.features_ = pd.MultiIndex.from_arrays([self.groups_ * X.shape[1], X.columns])
        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Group the columns of the input DataFrame X according to the MultiIndex.

        Parameters
        ----------
        X : pd.DataFrame
           The input DataFrame to be transformed.

        Returns
        -------
        X_grouped : pd.DataFrame
           The input DataFrame with its columns grouped according to the MultiIndex.
        """

        # Copy the input DataFrame if specified, and assign the MultiIndex to the columns.
        X = X.copy() if self.copy else X
        X.columns = self.features_
        return X


class FunctionTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a given function to each element of a pandas DataFrame or NumPy array.
    This transformer can also apply an inverse function to the transformed data.

    Parameters
    ----------
    func : callable
        The function to apply to the data.
    inverse_func : callable, optional (default=None)
        The inverse function to apply to the transformed data.
        If None, the transformed data is returned unchanged.

    Attributes
    ----------
    func : callable
        The function to apply to the data.
    inverse_func : callable
        The inverse function to apply to the transformed data.
    """
    def __init__(self,
                 func: Optional[Callable] = None,
                 inverse_func: Optional[Callable] = None):
        self.inverse_func = inverse_func
        self.func = func

    def fit(self) -> 'FunctionTransformer':
        """
        No-op method that returns self.
        """
        pass

    def transform(self,
                  X: pd.DataFrame) -> Union[pd.DataFrame, np.array]:
        """
        Applies the function to each element of the input data X.
        Returns the transformed data.

        Parameters
        ----------
        X : pandas DataFrame or NumPy array
            The input data to transform.

        Returns
        -------
        transformed : pandas DataFrame or NumPy array
            The transformed data.
        """

        # check if the input data is a pandas DataFrame
        if isinstance(X, pd.DataFrame):
            # apply the function to each element of the DataFrame
            return X.apply(self.func)

        else:
            # apply the function to each element of the NumPy array
            return np.vectorize(self.func)(X)

    def inverse_transform(self,
                          X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the inverse function to each element of the transformed data X.
        Returns the original data.

        Parameters
        ----------
        X : pandas DataFrame or NumPy array
            The transformed data to inverse transform.

        Returns
        -------
        original : pandas DataFrame or NumPy array
            The original data before transformation.
        """

        # check if an inverse function is provided
        if self.inverse_func is not None:

            # check if the input data is a pandas DataFrame
            if isinstance(X, pd.DataFrame):
                # apply the inverse function to each element of the DataFrame
                return X.apply(self.inverse_func)

            else:
                # apply the inverse function to each element of the NumPy array
                return np.vectorize(self.inverse_func)(X)

        else:
            # if no inverse function is provided, return the transformed data unchanged
            return X
