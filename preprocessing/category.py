import pandas as pd
import numpy as np

from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing

from pandas.api.types import CategoricalDtype
from category_encoders import *

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.model_selection import check_cv
from sklearn import preprocessing, impute

#import dask_ml.preprocessing.OneHotEncoder

from robusta.utils import all_subsets

from .base import TypeSelector, ColumnSelector




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
        self.ohe = preprocessing.OneHotEncoder(**self.params)
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




class TargetEncoder(TargetEncoder):

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_missing='value', handle_unknown='value', min_samples_leaf=1,
                 smoothing=1.0):

        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.ordinal_encoder = None
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing # for python 3 only
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self._mean = None
        self.feature_names = None




class FastEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, smoothing=30.0):
        self.smoothing = smoothing


    def fit(self, X, y):

        # Fit Smoothed Likelihood Encoder to independent column
        col_encoder = lambda x: smoothed_likelihood(x, y, self.smoothing)

        # Save pre-fitted Encoders (key: column name)
        self.encoders = {col: col_encoder(x) for col, x in X.iteritems()}

        return self


    def transform(self, X):

        # Apply Encoders to each column
        return X.aggregate(lambda x: x.map(self.encoders[x.name]))




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




class EncoderCV(BaseEstimator):
    """Cross Encoder for supervised encoders.

    Parameters
    ----------
    encoder : encoder instance, default=TargetEncoder()

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

            - None, to use the default 3-fold cross-validation,
            - integer, to specify the number of folds.
            - CV splitter,
            - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if classifier is True and y is either binary or
        multiclass, StratifiedKFold is used. In all other cases, KFold is used.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel for both fit and transform. None
        means 1 unless in a joblib.parallel_backend context. -1 means using all
        processors.

    verbose : int, optional (default=0)
        Controls the verbosity level.

    """
    def __init__(self, encoder=FastEncoder(), cv=4, n_jobs=-1, verbose=0):
        self.encoder = encoder
        self.cv = cv

        self.verbose = verbose
        self.n_jobs = n_jobs


    def fit(self, X, y):

        self._pre_fit(X, y)

        jobs = (delayed(self._path_fit)(clone(self.encoder), X, y, trn)
                for trn, oof in self.folds)
        paths = Parallel(backend='multiprocessing', max_nbytes='256M', pre_dispatch='all',
                         n_jobs=self.n_jobs, verbose=self.verbose)(jobs)

        self.encoders = paths
        return self


    def transform(self, X, is_train_set=None):

        if is_train_set is None and hasattr(self, 'train_shape_'):
            is_train_set = self._check_identity(X)
        else:
            is_train_set = False

        if is_train_set:
            # In case if user directly tells that it is train set but shape is different
            if self.train_shape_ != X.shape:
                raise ValueError('Train set must have the same shape '
                                 'in order to be transformed.')

            # If everything is OK, get out-of-fold predictions
            jobs = (delayed(self._path_enc)(encoder, X, oof=self.folds[i][1])
                    for i, encoder in enumerate(self.encoders))

        else:
            # For test set just averaging all predictions
            jobs = (delayed(self._path_enc)(encoder, X)
                    for i, encoder in enumerate(self.encoders))

        paths = Parallel(backend='multiprocessing', max_nbytes='256M', pre_dispatch='all',
                         n_jobs=self.n_jobs, verbose=self.verbose)(jobs)

        preds = paths
        return self._mean_preds(preds)


    def fit_transform(self, X, y):

        self._pre_fit(X, y, footprint=False)

        jobs = (delayed(self._path_fit_enc)(clone(self.encoder), X, y, trn, oof)
                for trn, oof in self.folds)
        paths = Parallel(backend='multiprocessing', max_nbytes='256M', pre_dispatch='all',
                         n_jobs=self.n_jobs, verbose=self.verbose)(jobs)

        self.encoders, preds = zip(*paths)
        return self._mean_preds(preds)



    def _mean_preds(self, preds):
        return pd.concat(preds, axis=1).groupby(level=0, axis=1).mean()


    def _path_fit(self, encoder, X, y, trn):
        return encoder.fit(X.iloc[trn], y.iloc[trn])


    def _path_enc(self, encoder, X, oof=None):
        if oof is None:
            return encoder.transform(X)
        else:
            return encoder.transform(X.iloc[oof])


    def _path_fit_enc(self, encoder, X, y, trn, oof):
        encoder.fit(X.iloc[trn], y.iloc[trn])
        return encoder, encoder.transform(X.iloc[oof])


    def _pre_fit(self, X, y, footprint=True):

        #self.encoder.set_params(**self.params)

        is_classifier = len(np.unique(y)) < 100 # hack
        cv = check_cv(self.cv, y, is_classifier)
        self.folds = list(cv.split(X, y))

        if footprint:
            self.train_footprint_ = self._get_footprint(X)
            self.train_shape_ = X.shape


    def _check_identity(self, X, rtol=1e-05, atol=1e-08, equal_nan=False):
        """Checks 2d numpy array or sparse matrix identity
        by its shape and footprint.
        """
        try:
            X = X.values
            # Check shape
            if X.shape != self.train_shape_:
                return False
            # Check footprint
            try:
                for coo in self.train_footprint_:
                    assert np.isclose(X[coo[0], coo[1]], coo[2], rtol=rtol, atol=atol,
                                      equal_nan=equal_nan)
                return True
            except AssertionError:
                return False

        except Exception:
            raise ValueError('Internal error. '
                             'Please save traceback and inform developers.')


    def _get_footprint(self, X, n_items=1000):
        """Selects ``n_items`` random elements from 2d numpy array or
        sparse matrix (or all elements if their number is less or equal
        to ``n_items``).
        """
        try:
            X = X.values
            footprint = []
            r, c = X.shape
            n = r * c
            # np.random.seed(0) # for development
            ids = np.random.choice(n, min(n_items, n), replace=False)

            for i in ids:
                row = i // c
                col = i - row * c
                footprint.append((row, col, X[row, col]))

            return footprint

        except Exception:
            raise ValueError('Internal error. '
                             'Please save traceback and inform developers.')



class NaiveBayesEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, smooth):
        self.smooth = smooth


    def transform(self, X):
        return X.multiply(self._r)


    def fit(self, X, y):
        self._r = sparse.csr_matrix(np.log(self._pr(X, y, 1) / self._pr(X, y, 0)))
        return self


    def _pr(self, X, y, val):
        prob = X[y == val].sum(0)
        return (prob + self.smooth) / ((y == val).sum() + self.smooth)



FastEncoderCV = lambda **params: EncoderCV(FastEncoder(), **params)
TargetEncoderCV = lambda **params: EncoderCV(TargetEncoder(), **params)
MEstimateEncoderCV = lambda **params: EncoderCV(MEstimateEncoder(), **params)
JamesSteinEncoderCV = lambda **params: EncoderCV(JamesSteinEncoder(), **params)
WOEEncoderCV = lambda **params: EncoderCV(WOEEncoder(), **params)




def smoothed_likelihood(x, y, smoothing, min_samples_leaf=1):
    prior = y.mean()
    stats = y.groupby(x).agg(['count', 'mean'])
    smoove = 1 / (1 + np.exp(-(stats['count'] - min_samples_leaf) / smoothing))
    likelihood = prior * (1 - smoove) + stats['mean'] * smoothing
    likelihood[stats['count'] == 1] = prior
    return likelihood
