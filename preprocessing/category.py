import pandas as pd
import numpy as np

from itertools import combinations
from numpy.linalg import svd

from dask_ml.preprocessing import OneHotEncoder, DummyEncoder, OrdinalEncoder
from sklearn.base import clone, BaseEstimator, TransformerMixin

from sklearn.utils.multiclass import type_of_target
from robusta.utils import all_subsets

from category_encoders import *


__all__ = [
    'LabelBinarizer',
    'OrdinalEncoder',
    'LabelEncoder1D',
    'LabelEncoder',
    'Categorizer1D',
    'Categorizer',
    'OneHotEncoder',
    'DummyEncoder',
    'FrequencyEncoder',
    'FeatureCombiner',
    'BackwardDifferenceEncoder',
    'BinaryEncoder',
    'HashingEncoder',
    'HelmertEncoder',
    'OrdinalEncoder',
    'SumEncoder',
    'PolynomialEncoder',
    'BaseNEncoder',
    'SVDEncoder',
    'ThermometerEncoder1D',
    'ThermometerEncoder',
    'GroupByEncoder',
]




class LabelEncoder1D(BaseEstimator, TransformerMixin):
    """Encode categories as integers.
    """
    def __init__(self):
        pass


    def fit(self, y):

        self.cats_ = y.astype('category').cat.categories
        self.dtype = y.dtype

        self.mapper = dict(zip(self.cats_, range(len(self.cats_))))
        self.inv_mapper = {val: key for key, val in self.mapper.items()}

        self.mapper[np.nan] = -1
        self.inv_mapper[-1] = np.nan

        return self


    def transform(self, y):
        return y.map(self.mapper)


    def inverse_transform(self, y):
        return y.map(self.inv_mapper).astype(self.dtype)




class LabelEncoder(LabelEncoder1D):

    def fit(self, X, y=None):

        self.transformers = {}
        for col in X.columns:
            self.transformers[col] = LabelEncoder1D().fit(X[col])

        return self


    def transform(self, X):
        Xt = pd.DataFrame(index=X.index, columns=X.columns)

        for col, transformer in self.transformers.items():
            Xt[col] = transformer.transform(X[col])

        return Xt


    def inverse_transform(self, X):

        Xt = pd.DataFrame(index=X.index, columns=X.columns)

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

        return Xt.astype(float)




class FeatureCombiner(BaseEstimator, TransformerMixin):
    """Extract Feature Combinations
    """
    def __init__(self, orders=[2, 3], sep=','):
        self.orders = orders
        self.sep = sep


    def fit(self, X, y=None):

        subsets = all_subsets(X.columns, self.orders)
        self.subsets_ = [list(subset) for subset in subsets]
        self.n_subsets_ = len(self.subsets_)

        return self


    def transform(self, X):

        X = X.astype(str)
        X = pd.concat([X[subset].apply(self.sep.join, axis=1).rename(sep(subset))
                      for subset in subsets], axis=1)
        return X




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



class LabelBinarizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, y):

        if len(y) == 0:
            raise ValueError(f"y has 0 samples: {y}")

        self.y_type_ = type_of_target(y)
        self.y_name_ = y.name

        if 'multioutput' in self.y_type_:
            raise ValueError("Multioutput target data is not supported")

        self.classes_ = y.astype('category').values.categories

        if len(self.classes_) == 1:
            raise ValueError(f"y has single class: {self.classes_}")

        elif len(self.classes_) == 2:
            self.mapper_     = dict(zip(self.classes_, [0, 1]))
            self.inv_mapper_ = dict(zip([0, 1], self.classes_))

        elif len(self.classes_) >= 3:
            y = pd.DataFrame(y)
            self.encoder_ = DummyEncoder().fit(y)

        else:
            raise ValueError(f"{self.y_type_} target data is not supported")

        return self


    def transform(self, y):

        if self.y_type_ is 'binary':
            y = y.map(self.mapper_).astype('uint8')

        elif self.y_type_ is 'multiclass':
            y = pd.DataFrame(y)
            y = self.encoder_.transform(y)
            y.columns = self.classes_

        return y


    def inverse_transform(self, y):

        if self.y_type_ is 'binary':
            y = y.map(self.inv_mapper_)

        elif self.y_type_ is 'multiclass':
            y = y.apply(lambda row: row.argmax(), axis=1)
            y.name = self.y_name_

        return y



class ThermometerEncoder1D(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, y):

        self.cats_ = y.astype('category').cat.categories
        self.type_ = y.dtype
        self.name_ = y.name

        return self


    def transform(self, y):
        y = pd.concat(map(lambda cat: cat <= y, self.cats_), axis=1)
        y.columns = self.cats_
        y.rename(lambda cat: f"{self.name_}:{cat}", axis=1, inplace=True)
        return y.astype('uint8')


    def inverse_transform(self, y):
        y = pd.Series(self.cats_[y.sum(axis=1)-1], index=y.index,
                      name=self.name_, dtype=self.type_)
        return y



class ThermometerEncoder(ThermometerEncoder1D):

    def fit(self, X, y=None):

        self.transformers = {}
        for col in X.columns:
            self.transformers[col] = ThermometerEncoder1D().fit(X[col])

        return self


    def transform(self, X):

        X_list = []
        for col, transformer in self.transformers.items():
            X_list.append(transformer.transform(X[col]))

        return pd.concat(X_list, axis=1)


    def inverse_transform(self, X):

        X_list = []
        for col, transformer in self.transformers.items():
            col_filter = ColumnFilter(lambda s: s.startswith(col))
            x = col_filter.fit_transform(X)
            x = transformer.inverse_transform(x)
            X_list.append(x)

        return pd.concat(X_list, axis=1)



class GroupByEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, func='mean', diff=False):
        self.func = func
        self.diff = diff

    def fit(self, X, y=None):
        self.cats_ = list(X.select_dtypes(['category', 'object']))
        self.nums_ = list(X.select_dtypes(np.number))
        return self

    def transform(self, X):
        Xt = pd.DataFrame(index=X.index)
        for cat in self.cats_:
            for num in self.nums_:
                col = num+'__'+cat
                Xt[col] = X[num].groupby(X[cat]).transform(self.func)
                if self.diff: Xt[col] = X[num] - Xt[col]

        return Xt
