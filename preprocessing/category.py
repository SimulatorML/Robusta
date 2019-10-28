import pandas as pd
import numpy as np

from itertools import combinations
from numpy.linalg import svd

from tqdm import tqdm

from sklearn.base import clone, BaseEstimator, TransformerMixin
import sklearn.preprocessing

from robusta.utils import all_subsets




class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a one-hot numeric array.
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    The input to this transformer should be an DataFrame of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array.

    By default, the encoder derives the categories based on the unique values
    in each feature. Alternatively, you can also specify the `categories`
    manually.

    Parameters
    ----------
    sep : string, default='_'
        Separator for column's name and its category.

    categories : 'auto' or a list of lists/arrays of values, default='auto'.
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values within a single feature, and should be sorted in case of
          numeric values.

        The used categories can be found in the ``categories_`` attribute.

    sparse : boolean, default=True
        Will return sparse matrix if set True else will return an array.

    dtype : number type, default=np.uint8
        Desired dtype of output.

    handle_unknown : 'error' or 'ignore', default='ignore'.
        Whether to raise an error or ignore if an unknown categorical feature
        is present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros. In the inverse transform, an unknown category
        will be denoted as None.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``).

    """
    def __init__(self, sep='_', **params):
        self.sep = sep
        self.params = params


    def fit(self, X, y=None):
        """Fit OneHotEncoder to X.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to determine the categories of each feature.

        Returns
        -------
        self

        """
        self.ohe = sklearn.preprocessing.OneHotEncoder(**self.params)
        self.ohe.fit(X, y)

        ohe_columns = self.ohe.get_feature_names()
        self.categories_ = self.ohe.categories_

        x_columns = X.columns
        self.columns = []
        for ohe_column in ohe_columns:
            col, cat = ohe_column.split('_', 1)
            col = x_columns[int(col[1:])]
            column = '{}{}{}'.format(col, self.sep, cat)
            self.columns.append(column)

        return self


    def transform(self, X):
        """Transform X using one-hot encoding.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to encode.

        Returns
        -------
        X_ohe : sparse DataFrame if sparse=True else simple DataFrame
            Transformed input.

        """

        X_ohe = self.ohe.transform(X)

        if self.ohe.sparse:
            X_ohe = pd.DataFrame.sparse.from_spmatrix(X_ohe)
        else:
            X_ohe = pd.DataFrame(X_ohe)

        X_ohe.columns = self.columns
        X_ohe.index = X.index

        return X_ohe




class LabelEncoder1D(BaseEstimator, TransformerMixin):
    """Encode categories as integers.
    """
    def __init__(self):
        pass


    def fit(self, y):
        """Fit LabelEncoder to y.

        Parameters
        ----------
        y : Series

        Returns
        -------
        self

        """
        self.cats_ = y.astype('category').values.categories
        self.dtype = y.dtype

        self.mapper = dict(zip(self.cats_, range(len(self.cats_))))
        self.inv_mapper = {val: key for key, val in self.mapper.items()}

        self.mapper[np.nan] = -1
        self.inv_mapper[-1] = np.nan

        return self


    def transform(self, y):
        """Transform y.

        Parameters
        ----------
        y : Series

        Returns
        -------
        yt : Series
            Transformed input.

        """
        return y.map(self.mapper)


    def inverse_transform(self, y):
        """Inverse transform y.

        Parameters
        ----------
        y : Series

        Returns
        -------
        yt : Series
            Inverse transformed input.

        """
        return y.map(self.inv_mapper).astype(self.dtype)




class LabelEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as integers.
    """
    def __init__(self):
        pass


    def fit(self, X, y=None):
        """Fit LabelEncoder to X.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to determine the categories of each feature.

        Returns
        -------
        self

        """
        self.transformers = {}
        for col in X.columns:
            self.transformers[col] = LabelEncoder1D().fit(X[col])

        return self


    def transform(self, X):
        """Transform X using label encoding.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        Xt : DataFrame, shape [n_samples, n_features]
            Transformed input.

        """
        Xt = pd.DataFrame(index=X.index)

        for col, transformer in self.transformers.items():
            Xt[col] = transformer.transform(X[col])

        return Xt


    def inverse_transform(self, X):
        """Inverse transform X using label encoding.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        Xt : DataFrame, shape [n_samples, n_features]
            Inverse transformed input.

        """
        Xt = pd.DataFrame(index=X.index)

        for col, transformer in self.transformers.items():
            Xt[col] = transformer.inverse_transform(X[col])

        return Xt




class Categorizer1D(BaseEstimator, TransformerMixin):
    """Convert categories to 'category' dtype of the same range.
    """
    def __init__(self):
        pass


    def fit(self, y):
        """Learn categories

        Parameters
        ----------
        y : Series

        Returns
        -------
        self

        """
        self.cats_ = y.astype('category').values.categories

        return self


    def transform(self, y):
        """Convert y to fitted categories

        Parameters
        ----------
        y : Series

        Returns
        -------
        yt : Series
            Transformed input.

        """
        return pd.Categorical(y, categories=self.cats_)




class Categorizer(BaseEstimator, TransformerMixin):
    """Convert categories to 'category' dtype of the same range.
    """
    def __init__(self):
        pass


    def fit(self, X, y=None):
        """Learn categories

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to determine the categories of each feature.

        Returns
        -------
        self

        """
        self.transformers = {}
        for col in X.columns:
            self.transformers[col] = Categorizer1D().fit(X[col])

        return self


    def transform(self, X):
        """Convert X to fitted categories

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        Xt : DataFrame, shape [n_samples, n_features]
            Transformed input.

        """
        Xt = pd.DataFrame(index=X.index)

        for col, transformer in self.transformers.items():
            Xt[col] = transformer.transform(X[col])

        return Xt




class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as it's frequencies.
    """
    def __init__(self, normalize=True):
        self.normalize = normalize


    def fit(self, X, y=None):
        """Fit FrequencyEncoder to X.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to determine frequencies.

        Returns
        -------
        self

        """
        norm = self.normalize
        self.value_counts_ = {col: x.value_counts(norm) for col, x in X.items()}

        return self


    def transform(self, X):
        """Transform X using frequency encoding.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        Xt : DataFrame, shape [n_samples, n_features]
            Transformed input.

        """
        Xt = pd.DataFrame(index=X.index)

        for col, vc in self.value_counts_.items():
            Xt[col] = X[col].map(vc)

        return Xt




class FeatureCombiner(BaseEstimator, TransformerMixin):
    """Extract Feature Combinations
    """
    def __init__(self, orders=[2, 3], sep=',', tqdm=False):
        self.orders = orders
        self.sep = sep
        self.tqdm = tqdm


    def fit(self, X, y=None):
        """Fit FeatureCombiner to X.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            Memorize columns

        Returns
        -------
        self

        """
        subsets = all_subsets(X.columns, self.orders)
        self.subsets_ = [list(subset) for subset in subsets]
        self.n_subsets_ = len(self.subsets_)

        return self


    def transform(self, X):
        """Transform X using FeatureCombiner

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        Xt : DataFrame, shape [n_samples, n_features]
            Transformed input.

        """
        X = X.astype(str)
        sep = self.sep.join

        subsets = tqdm(self.subsets_) if self.tqdm else self.subsets_

        Xt = pd.concat([X[subset].apply(sep, axis=1).rename(sep(subset))
                        for subset in subsets], axis=1)

        return Xt




class SVDEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features by pairwise transforming
    categorical features to the counter matrix
    and embedding with SVD.

    """
    def __init__(self, n_components=0.9):
        self.n_components = n_components


    def fit(self, X, y=None):
        """Fit data

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to determine frequencies.

        Returns
        -------
        self

        """

        # Check data
        assert not X.isna().any().any(), 'Missing values are not allowed'

        columns = X.columns
        self.embeddings_ = {col: pd.DataFrame(index=X[col].unique()) for col in columns}

        self.n_components_ = pd.DataFrame(index=columns, columns=columns)
        self.sigmas_ = {}

        for a, b in combinations(columns, 2):

            # Count Matrix
            x = X.groupby([a, b]).size().unstack().fillna(0)

            # SVD
            u, s, v = svd(x, full_matrices=False)
            v = v.T

            # n_components
            if isinstance(self.n_components, int):
                n_components_ = min(self.n_components, len(s))

            elif isinstance(self.n_components, float):
                ratio = s.cumsum()/s.sum()
                n_components_ = (ratio > self.n_components).argmax() + 1

            else:
                raise ValueError('Unknown n_components type:', self.n_components)

            self.n_components_[a, b] = n_components_
            self.n_components_[b, a] = n_components_

            # Truncate
            u_cols, v_cols = [], []

            for i in range(n_components_):
                u_cols.append('({},{})_svd{}'.format(a, b, i+1))
                v_cols.append('({},{})_svd{}'.format(b, a, i+1))

            u = pd.DataFrame(u[:, :n_components_], columns=u_cols, index=x.index)
            v = pd.DataFrame(v[:, :n_components_], columns=v_cols, index=x.columns)

            # Append to Embeddings
            self.embeddings_[a] = self.embeddings_[a].join(u)
            self.embeddings_[b] = self.embeddings_[b].join(v)

        return self


    def transform(self, X):
        """Transform data

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        Xt : DataFrame, shape [n_samples, n_features]
            Transformed input.

        """
        return pd.concat([self.embeddings_[col].loc[x].set_index(x.index)
                          for col, x in X.items()], axis=1)
