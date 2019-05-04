import pandas as pd
import numpy as np

from joblib import Parallel, delayed
import multiprocessing

from pandas.api.types import CategoricalDtype
from category_encoders import *

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.model_selection import check_cv
from sklearn import preprocessing, impute

from .basic import TypeSelector, ColumnSelector




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
    def __init__(self, sep='_', categories='auto', sparse=True, dtype=np.uint8,
                 handle_unknown='ignore'):
        self.sep = sep
        self.categories = categories
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown


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
        self.ohe = preprocessing.OneHotEncoder(sparse=self.sparse, dtype=self.dtype,
            handle_unknown=self.handle_unknown, categories=self.categories)
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
        if self.sparse:
            return pd.SparseDataFrame(X_ohe, columns=self.columns, index=X.index)
        else:
            return pd.DataFrame(X_ohe, columns=self.columns, index=X.index)




class LabelEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as integers.

    Parameters
    ----------
    dtype : number type, default=np.uint8
        Desired dtype of output.

    Attributes
    ----------
    categories_ : dict of lists
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``). The keys of dictionary is column names, the values
        are lists of categories.

    """
    def __init__(self, dtype=np.uint8):
        self.dtype = dtype


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
        self.categories_ = {}
        for col in X.columns:
            self.categories_[col] = X[col].astype('category').cat.categories

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

        for col, cats in self.categories_.items():
            cat_dtype = CategoricalDtype(cats, ordered=False)
            Xt[col] = X[col].astype(cat_dtype).cat.codes

        return Xt.astype(self.dtype)



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




class EncoderCV(BaseEstimator):
    """Cross Encoder for supervised encoders.

    Parameters
    ----------
    dtype : number type, default=np.uint8
        Desired dtype of output.

    Attributes
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
        The number of jobs to run in parallel for both fit and predict. None
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


    #def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.

        """
        #return self._get_params('encoders', deep=deep)


    #def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self

        """
        #self._set_params('encoders', **kwargs)
        #return self



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
