import collections
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from dask_ml.preprocessing import DummyEncoder
from numpy.linalg import svd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target

from . import ColumnFilter
from ..utils import all_subsets


class LabelEncoder1D(BaseEstimator, TransformerMixin):
    """
    LabelEncoder1D is a transformer that encodes a 1D array of categorical labels
    as integers. The encoder can also reverse the encoding to map the integers back
    to the original labels.

    Parameters
    ----------
    None

    Attributes
    ----------
    inv_mapper : dict
        A dictionary mapping integer-encoded labels to their original categorical labels.
    mapper : dict
        A dictionary mapping categorical labels to their integer encodings.
    dtype : numpy.dtype
        The dtype of the input array.
    cats_ : numpy.ndarray
        The unique categorical labels in the input array.
    """
    def __init__(self):
        self.inv_mapper = None
        self.mapper = None
        self.dtype = None
        self.cats_ = None

    def fit(self,
            y: pd.Series) -> 'LabelEncoder1D':
        """
        Fit the encoder to the input array of categorical labels.

        Parameters
        ----------
        y : pandas.Series
            A array of categorical labels.

        Returns
        -------
        self : LabelEncoder1D
            Returns the encoder.
        """

        # Calculate the unique categorical labels in the input array and assign them to the cats_ attribute.
        self.cats_ = np.unique(y)

        # Determine the dtype of the input array and assign it to the dtype attribute.
        self.dtype = y.dtype

        # Create a dictionary mapping each categorical label to an integer encoding and assign it to the mapper
        # attribute. Also create an inverse mapping from the integer encodings to the categorical labels and assign
        # it to the inv_mapper attribute.
        self.mapper = dict(zip(self.cats_, range(len(self.cats_))))
        self.inv_mapper = {val: key for key, val in self.mapper.items()}

        # Add a special case for NaN values in the input array by mapping them to -1.
        # The inverse mapping will map -1 back to NaN.
        self.mapper[np.nan] = -1
        self.inv_mapper[-1] = np.nan

        return self

    def transform(self,
                  y: pd.Series) -> pd.Series:
        """
        Transform an input array of categorical labels to integer encodings.

        Parameters
        ----------
        y : pandas.Series
            A array of categorical labels.

        Returns
        -------
        y_enc : pandas.Series
            An array of integer encodings.
        """

        # Map each categorical label to its corresponding integer encoding using the mapper attribute.
        return y.map(self.mapper)

    def inverse_transform(self,
                          y: pd.Series) -> pd.Series:
        """
        Transform an input array of integer encodings back to their original categorical labels.

        Parameters
        ----------
        y : pandas.Series
            A array of integer encodings.

        Returns
        -------
        y_orig : pandas.Series
            A array of categorical labels.
        """

        # Map the integer encodings to their original categorical labels using the inv_mapper dictionary
        # and convert the resulting series to the original data type of the input array
        return y.map(self.inv_mapper).astype(self.dtype)


class LabelEncoder:
    """
    Encodes categorical features in a pandas DataFrame using LabelEncoder1D.

    Attributes
    ----------
    transformers : dict
        A dictionary containing a LabelEncoder1D transformer for each column in the input DataFrame.

    """
    def __init__(self):
        self.transformers = None

    def fit(self,
            X: pd.DataFrame) -> 'LabelEncoder':
        """
        Fits a LabelEncoder1D instance to each column of the input pandas DataFrame.

        Parameters
        ----------
        X : pandas DataFrame
            The input DataFrame to fit.

        Returns
        -------
        self : LabelEncoder
            The fitted LabelEncoder instance.
        """

        # create a dictionary to hold transformers for each column
        self.transformers = {}

        # loop through each column in the input DataFrame
        for col in X.columns:
            # create a new LabelEncoder1D instance and fit it to the current column
            self.transformers[col] = LabelEncoder1D().fit(X[col])
        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a pandas DataFrame using the fitted LabelEncoder1D instances.

        Parameters
        ----------
        X : pandas DataFrame
            The input DataFrame to transform.

        Returns
        -------
        Xt : pandas DataFrame
            The transformed DataFrame.
        """

        # make a copy of the input DataFrame
        Xt = X.copy()

        # loop through each column and apply the corresponding LabelEncoder1D transformer
        for col, transformer in self.transformers.items():
            Xt[col] = transformer.transform(X[col])
        return Xt

    def inverse_transform(self,
                          X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transforms a pandas DataFrame using the fitted LabelEncoder1D instances.

        Parameters
        ----------
        X : pandas DataFrame
            The input DataFrame to inverse transform.

        Returns
        -------
        Xt : pandas DataFrame
            The inverse transformed DataFrame.
        """

        # make a copy of the input DataFrame
        Xt = X.copy()

        # loop through each column and apply the corresponding LabelEncoder1D inverse transformer
        for col, transformer in self.transformers.items():
            Xt[col] = transformer.inverse_transform(X[col])
        return Xt


class Categorizer1D(BaseEstimator, TransformerMixin):
    """
    Transformer to convert a pandas Series into a categorical variable with pre-defined categories.

    This transformer infers the unique categories from the input data during the `fit` method, and stores them in
    the `cats_` attribute. The `transform` method then converts the input data into a categorical variable using
    the categories stored in the `cats_` attribute.

    Attributes
    ----------
    cats_ : pandas CategoricalIndex or None
        The unique categories inferred from the input data during the `fit` method. If `fit` has not yet been called,
        this attribute is None.
    """
    def __init__(self):
        self.cats_ = None

    def fit(self,
            y: pd.Series) -> 'Categorizer1D':
        """
        Infers the unique categories from the input y and stores them in `cats_` attribute.

        Parameters
        ----------
        y: pd.Series
            The input data to be transformed into categorical variable.

        Returns
        -------
        self: Categorizer1D
            Returns the instance of the Categorizer1D class with `cats_` attribute set to unique categories.
        """

        # infer unique categories from input y and store them in `cats_` attribute
        self.cats_ = y.astype('category').values.categories
        return self

    def transform(self,
                  y: pd.Series) -> pd.Categorical:
        """
        Converts the input y into a categorical variable using the categories stored in `cats_` attribute.

        Parameters
        ----------
        y: pd.Series
            The input data to be transformed into categorical variable.

        Returns
        -------
        cat_y: Categorical
            Returns the transformed categorical variable with categories specified in `cats_` attribute.
        """

        # convert input y into categorical variable using the categories stored in `cats_` attribute
        return pd.Categorical(y, categories=self.cats_)


class Categorizer(BaseEstimator, TransformerMixin):
    """
    Transformer for converting categorical columns in a pandas DataFrame into numerical values.

    This transformer fits a separate Categorizer1D transformer on each column of the input DataFrame.
    The Categorizer1D transformer maps each unique value in a column to a unique integer.

    Attributes
    ----------
    transformers : dict
        A dictionary mapping column names to the fitted Categorizer1D transformers.
    """
    def __init__(self):
        self.transformers = None

    def fit(self,
            X: pd.DataFrame) -> 'Categorizer':
        """
        Fit a separate Categorizer1D transformer on each column of the input DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to fit the transformers on.

        Returns
        -------
        self:
            The fitted transformer object.
        """

        # initialize the transformers dictionary
        self.transformers = {}

        # loop through each column in the input DataFrame
        for col in X.columns:
            # fit a Categorizer1D transformer on the current column and store it in the transformers dictionary
            self.transformers[col] = Categorizer1D().fit(X[col])

        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by applying the fitted Categorizer1D transformers.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to transform.

        Returns
        -------
        Xt : pd.DataFrame
            The transformed DataFrame.
        """

        # create an empty DataFrame to hold the transformed values
        Xt = pd.DataFrame(index=X.index)

        # loop through each column and apply the corresponding Categorizer1D transformer
        for col, transformer in self.transformers.items():
            Xt[col] = transformer.transform(X[col])

        return Xt


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    A transformer that encodes categorical features using the frequency of each category in the training set.

    Parameters:
    -----------
    normalize : bool, default=True
        Whether to normalize the encoded values to be between 0 and 1.

    Attributes:
    -----------
    value_counts_ : dict
        A dictionary containing the value counts for each categorical feature in the training set.
    """
    def __init__(self,
                 normalize: bool = True):
        self.value_counts_ = None
        self.normalize = normalize

    def fit(self,
            X: pd.DataFrame) -> 'FrequencyEncoder':
        """
        Fit the transformer to the training data.

        Parameters
        ----------
        X : pandas DataFrame
            The training data to fit the transformer to.

        Returns
        -------
        self : FrequencyEncoder
            The fitted transformer.
        """

        # Compute the value counts for each column in X and store them in a dictionary
        self.value_counts_ = {col: collections.Counter(X[col]) for col in X.columns}
        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by encoding the categorical features using their frequency in the training set.

        Parameters
        ----------
        X : pandas DataFrame
            The input data to encode.

        Returns
        -------
        Xt : pandas DataFrame
            The encoded data.
        """

        # Map each categorical value to its frequency in the training set
        Xt = X.apply(lambda col: col.map(self.value_counts_[col.name]))

        # Convert the encoded data to a float type
        return Xt.astype(float)


class FeatureCombiner(BaseEstimator, TransformerMixin):
    """
    Transformer that combines features by taking all possible subsets of features
    up to a certain order and concatenating their values using a separator.

    Parameters:
    ----------
    orders: list, optional (default=[2, 3])
        List of integers representing the orders of the feature subsets to be considered.
        For example, orders=[2, 3] will consider subsets of features of orders 2 and 3.
    sep: str, optional (default=',')
        Separator used to concatenate the values of the features in each subset.

    Attributes:
    ----------
    subsets_: list
        List of all possible subsets of features up to the highest order specified in orders.
    n_subsets_: int
        Total number of subsets generated.
    """
    def __init__(self,
                 orders: Optional[list] = None,
                 sep: str = ','):
        self.n_subsets_ = None
        self.subsets_ = None
        if orders is None:
            orders = [2, 3]
        self.orders = orders
        self.sep = sep

    def fit(self,
            X: pd.DataFrame) -> 'FeatureCombiner':
        """
        Generate all possible subsets of features up to the highest order specified in orders.

        Parameters
        ----------
        X: pandas DataFrame
            Input data containing the features.

        Returns
        -------
        self: FeatureCombiner
            Returns self.
        """

        # Generate all possible subsets of features up to the highest order specified in orders
        subsets = all_subsets(X.columns, self.orders)
        self.subsets_ = [list(subset) for subset in subsets]
        self.n_subsets_ = len(self.subsets_)

        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Combine features by taking all possible subsets of features and concatenating their
        values using the separator specified in sep.

        Parameters
        ----------
        X: pandas DataFrame
            Input data containing the features.

        Returns
        -------
        X: pandas DataFrame
            Transformed features where each feature is the concatenation of all possible subsets
            of features up to the highest order specified in orders.
        """

        # Convert input data to string
        X = X.astype(str)

        # Concatenate values of all possible subsets of features up to the highest order specified in orders
        X = pd.concat([X[subset].apply(self.sep.join, axis=1).rename(sep(subset))
                       for subset in self.subsets_], axis=1)
        return X


class SVDEncoder(BaseEstimator, TransformerMixin):
    """
    Encode a pandas DataFrame using Singular Value Decomposition (SVD).

    For each pair of columns in the input DataFrame, the count matrix of their joint values is computed
    and then decomposed using SVD. The resulting left and right singular vectors are used to embed
    the unique values of each column into a lower-dimensional space. The dimensionality of this space
    is determined either by specifying a fixed number of components or by specifying a cumulative
    variance ratio threshold.

    Parameters
    ----------
    n_components : int or float, default=0.9
        If int, specifies the fixed number of components to use for all pairs of columns.
        If float, specifies the cumulative variance ratio threshold to use for all pairs of columns.
        In this case, the number of components for each pair of columns is determined by the smallest
        number of components needed to reach the specified threshold.

    Attributes
    ----------
    embeddings_ : dict
        A dictionary where each key is the name of a column in the input DataFrame and each value
        is a DataFrame containing the embeddings of the unique values of that column.
    n_components_ : pd.DataFrame
        A square DataFrame where the row and column labels are the names of columns in the input
        DataFrame. The (i, j) entry contains the number of components used to embed the unique
        values of columns i and j.
    sigmas_ : dict
        A dictionary where each key is a pair of column names and each value is a 1D array containing
        the singular values of the count matrix for that pair of columns.
    """

    def __init__(self,
                 n_components: float = 0.9):
        self.sigmas_ = None
        self.n_components_ = None
        self.embeddings_ = None
        self.n_components = n_components

    def fit(self,
            X: pd.DataFrame) -> 'SVDEncoder':
        """
        Compute the SVD embeddings for each pair of columns in the input DataFrame X.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to encode. Must not contain missing values.

        Returns
        -------
        self : SVDEncoder
            Returns self.

        """

        # Check for missing values in input DataFrame
        assert not X.isna().any().any(), 'Missing values are not allowed'

        # Initialize embeddings dictionary, number of components dataframe, and sigmas dictionary
        self.embeddings_ = {}
        self.n_components_ = pd.DataFrame(index=X.columns, columns=X.columns)
        self.sigmas_ = {}

        # Compute SVD embeddings for each pair of columns
        for a, b in combinations(X.columns, 2):
            # Compute count matrix for pairs of columns a and b
            x = X.groupby([a, b]).size().unstack().fillna(0)

            # Compute SVD of count matrix
            u, s, v = svd(x, full_matrices=False)

            # Determine number of components to use for SVD based on n_components
            if isinstance(self.n_components, int):
                n_components_ = min(self.n_components, len(s))
            elif isinstance(self.n_components, float):
                ratio = s.cumsum() / s.sum()
                n_components_ = (ratio > self.n_components).argmax() + 1
            else:
                raise ValueError('Unknown n_components type:', self.n_components)

            # Save number of components used for a pair of columns a and b
            self.n_components_.loc[a, b] = n_components_
            self.n_components_.loc[b, a] = n_components_

            # Create column names for left and right singular vectors
            u_cols = [f'({a},{b})_svd{i}' for i in range(1, n_components_ + 1)]
            v_cols = [f'({b},{a})_svd{i}' for i in range(1, n_components_ + 1)]

            # Create DataFrame of left singular vectors and add to embeddings dictionary
            u = pd.DataFrame(u[:, :n_components_], columns=u_cols, index=x.index)
            v = pd.DataFrame(v[:, :n_components_], columns=v_cols, index=x.columns)

            # Create DataFrame of right singular vectors and add to embeddings dictionary
            self.embeddings_[a] = self.embeddings_.get(a, pd.DataFrame(index=X[a].unique())).join(u)
            self.embeddings_[b] = self.embeddings_.get(b, pd.DataFrame(index=X[b].unique())).join(v)

        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by mapping each column to its SVD embeddings.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to be transformed.

        Returns
        -------
        pandas.DataFrame
            The transformed data with each column mapped to its SVD embeddings.
        """

        # Create a DataFrame of SVD embeddings for each column in input DataFrame
        return pd.concat([self.embeddings_[col].loc[x].set_index(x.index)
                          for col, x in X.items()], axis=1)


class LabelBinarizer(BaseEstimator, TransformerMixin):
    """
    Transformer to convert a categorical target variable to binary or one-hot encoded format.
    """
    def __init__(self):
        self.y_type_ = None
        self.y_name_ = None
        self.classes_ = None
        self.mapper_ = None
        self.inv_mapper_ = None
        self.encoder_ = None

    def fit(self,
            y: pd.Series) -> 'LabelBinarizer':
        """
        Fit the transformer to the target variable.

        Parameters
        ----------
        y : pd.Series
            The target variable to be transformed.

        Returns
        -------
        self : LabelBinarizer
            The fitted transformer.
        """

        if len(y) == 0:
            raise ValueError(f"y has 0 samples: {y}")

        # Determine the type of the target variable
        self.y_type_ = type_of_target(y)
        self.y_name_ = y.name

        # Raise an error if multioutput target data is provided
        if 'multioutput' in self.y_type_:
            raise ValueError("Multioutput target data is not supported")

        # Get the unique classes in the target variable
        self.classes_ = y.astype('category').values.categories

        # Raise an error if there is only one class in the target variable
        if len(self.classes_) == 1:
            raise ValueError(f"y has single class: {self.classes_}")

        # If there are two classes, create a mapping dictionary for binary encoding
        elif len(self.classes_) == 2:
            self.mapper_ = dict(zip(self.classes_, [0, 1]))
            self.inv_mapper_ = dict(zip([0, 1], self.classes_))

        # If there are three or more classes, use one-hot encoding
        elif len(self.classes_) >= 3:
            self.encoder_ = DummyEncoder().fit(y.values.reshape(-1, 1))

        # Raise an error if the target data type is not supported
        else:
            raise ValueError(f"{self.y_type_} target data is not supported")

        return self

    def transform(self,
                  y: pd.Series) -> pd.Series:
        """
        Transform the target variable to binary or one-hot encoded format.

        Parameters
        ----------
        y : pd.Series
            The target variable to be transformed.

        Returns
        -------
        y : pd.Series
            The transformed target variable.
        """

        # If binary encoding is used, map the class labels to binary values
        if self.y_type_ == 'binary':
            y = np.asarray([self.mapper_[c] for c in y])

        # If multiclass encoding is used, transform the target variable using the encoder
        elif self.y_type_ == 'multiclass':
            y = self.encoder_.transform(y.values.reshape(-1, 1)).toarray()
        return y

    def inverse_transform(self,
                          y: pd.Series) -> pd.Series:
        """
        Transform the binary or one-hot encoded target variable back to the original format.

        Parameters
        ----------
        y : pd.Series
            The transformed target variable.

        Returns
        -------
        y : pd.Series
            The original target variable.
        """

        # If binary encoding is used, map the binary values to class labels
        if self.y_type_ == 'binary':
            y = np.asarray([self.inv_mapper_[c] for c in y])

        # If multiclass encoding is used, transform the multiclass values to class labels
        elif self.y_type_ == 'multiclass':
            y = np.argwhere(y)[:, 1]
            y = pd.Series(self.classes_[y], name=self.y_name_)
        return y


class ThermometerEncoder1D(BaseEstimator, TransformerMixin):
    """
    One-dimensional thermometer encoder for categorical variables.

    This transformer takes a categorical variable and converts it to a one-dimensional thermometer encoding.
    A thermometer encoding is a binary encoding where each column corresponds to a category and contains a
    1 if the value is less than or equal to that category and 0 otherwise.
    """
    def __init__(self):
        self.type_ = None
        self.name_ = None
        self.cats_ = None

    def fit(self,
            y: pd.Series) -> 'ThermometerEncoder1D':
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        y : pandas.Series
            The input data to fit the transformer to.

        Returns
        -------
        self : ThermometerEncoder1D
            The fitted transformer.
        """

        # Convert the input data to a categorical variable
        self.cats_ = y.astype('category').cat.categories

        # Save the original data type, column name, and categories
        self.type_ = y.dtype
        self.name_ = y.name

        return self

    def transform(self,
                  y: pd.Series) -> pd.DataFrame:
        """
        Transform the input data to a one-dimensional thermometer encoding.

        Parameters
        ----------
        y : pandas.Series
            The input data to transform.

        Returns
        -------
        y : pandas.DataFrame
            The transformed data, with each column corresponding to a category and containing a 1 if the
            value is less than or equal to that category and 0 otherwise.
        """

        # Create a thermometer encoding by checking if each value is less than or equal to each category
        y = pd.concat(map(lambda cat: cat <= y, self.cats_), axis=1)

        # Set the column names to the category names
        y.columns = self.cats_

        # Rename the columns to include the original column name
        y.rename(lambda cat: f"{self.name_}:{cat}", axis=1, inplace=True)

        # Convert the data type to unsigned 8-bit integers
        return y.astype('uint8')

    def inverse_transform(self,
                          y: pd.DataFrame) -> pd.Series:
        """
        Inverse transform the one-dimensional thermometer encoding back to the original categorical variable.

        Parameters
        ----------
        y : pandas.DataFrame
            The one-dimensional thermometer encoded data to transform.

        Returns
        -------
        y : pandas.Series
            The inverse transformed data, with the same data type and column name as the original data.
        """

        # Compute the original categorical variable by counting the number of 1s in each row and mapping to the
        # category names
        y = pd.Series(self.cats_[y.sum(axis=1) - 1], index=y.index,
                      name=self.name_, dtype=self.type_)
        return y


class ThermometerEncoder(ThermometerEncoder1D):
    """
    Encodes categorical features using the Thermometer Encoding technique in a one-hot fashion.
    Inherits from ThermometerEncoder1D class for encoding individual features.
    """
    def __init__(self):
        super().__init__()
        self.transformers = None

    def fit(self,
            X: pd.DataFrame,
            y: Optional[pd.Series] = None) -> 'ThermometerEncoder':
        """
        Fit a thermometer encoder for each column in X.

        Parameters
        ----------
        X : pd.DataFrame
            The data to fit.
        y : pd.Series
            ignored

        Returns
        -------
        self : ThermometerEncoder
             The fitted transformer.
        """

        # Dictionary to store ThermometerEncoder1D objects for each column of X
        self.transformers = {}

        # Loop over each column in X and fit a ThermometerEncoder1D object for that column
        for col in X.columns:
            self.transformers[col] = ThermometerEncoder1D().fit(X[col])
        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform each categorical feature in X using the fitted thermometer encoders.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        dataframe : pd.DataFrame
            The transformed data.
        """

        # Concatenate the transformed data from each column's ThermometerEncoder1D object
        return pd.concat([self.transformers[col].transform(X[col]) for col in X.columns], axis=1)

    def inverse_transform(self,
                          X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the encoded data X back to its original form.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_encoded_features]
            The encoded data to transform back.

        Returns
        -------
        dataframe : pd.DataFrame
            The transformed data.
        """

        # Concatenate the inverse transformed data from each column's ThermometerEncoder1D object
        # Use ColumnFilter to select only the relevant columns from X
        # and then transform each column using the corresponding ThermometerEncoder1D object
        return pd.concat(
            [self.transformers[col].inverse_transform(ColumnFilter(lambda s: s.startswith(col)).fit_transform(X)) for
             col in X.columns], axis=1)


class GroupByEncoder(BaseEstimator, TransformerMixin):
    """
    Custom Transformer that groups numerical features by categorical features
    and applies an aggregation function to the groups.

    Parameters
    ----------
    func : str or function, default='mean'
        Aggregation function to apply to the groups. Can be a string ('mean', 'median', 'sum', 'std', 'min', 'max', etc.)
        or a callable function that takes in a pandas Series and returns a scalar.
    diff : bool, default=False
        Whether to subtract the aggregated value from the original value, resulting in the difference.

    Attributes
    ----------
    nums_ : list of str
        Names of the numerical features in the input DataFrame.
    cats_ : list of str
        Names of the categorical features in the input DataFrame.
    func : str or function
        The aggregation function to apply.
    diff : bool
        Whether to compute the difference between the original value and the aggregated value.
    """
    def __init__(self,
                 func: str = 'mean',
                 diff: bool = False):
        self.nums_ = None
        self.cats_ = None
        self.func = func
        self.diff = diff

    def fit(self,
            X: pd.DataFrame) -> 'GroupByEncoder':
        """
        Fit the Transformer to the input DataFrame.

        Parameters
        ----------
        X : pandas DataFrame
            Input DataFrame to fit the Transformer to.

        Returns
        -------
        self : GroupByEncoder
            This instance of the GroupByEncoder Transformer.
        """

        # Find the names of the categorical and numerical features in the input DataFrame.
        self.cats_ = list(X.select_dtypes(['category', 'object']))
        self.nums_ = list(X.select_dtypes(np.number))
        return self

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by grouping numerical features by categorical features
        and applying the aggregation function to the groups.

        Parameters
        ----------
        X : pandas DataFrame
            Input DataFrame to transform.

        Returns
        -------
        Xt : pandas DataFrame
            Transformed DataFrame with aggregated numerical features.
        """

        # Create an empty DataFrame with the same index as the input DataFrame.
        Xt = pd.DataFrame(index=X.index)
        grouped_data = {}

        # For each categorical feature and numerical feature pair:
        for cat, num in [(cat, num) for cat in self.cats_ for num in self.nums_]:

            # Create a new column name for the transformed feature.
            col = f"{num}__{cat}"

            # If we haven't seen this categorical feature before, group the data by it.
            if cat not in grouped_data:
                grouped_data[cat] = X.groupby(cat)

            # Apply the aggregation function to the numerical feature within each group of the categorical feature.
            Xt[col] = grouped_data[cat][num].transform(self.func)

            # If the 'diff' flag is set, compute the difference between the original value and the aggregated value.
            if self.diff:
                Xt[col] = X[num] - Xt[col]

        # Concatenate the transformed columns in alphabetical order by column name and return the resulting DataFrame.
        return pd.concat([Xt[col] for col in sorted(Xt.columns)], axis=1)
