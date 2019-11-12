import pandas as pd
import numpy as np

from itertools import combinations
from numpy.linalg import svd

from tqdm import tqdm

from sklearn.base import clone, BaseEstimator, TransformerMixin
import sklearn.preprocessing
import dask_ml.preprocessing

from robusta.utils import all_subsets




class OneHotEncoder(dask_ml.preprocessing.OneHotEncoder):
    pass


class DummyEncoder(dask_ml.preprocessing.DummyEncoder):
    pass


class OrdinalEncoder(dask_ml.preprocessing.OrdinalEncoder):
    pass




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

        return Xt.astype(float)




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
