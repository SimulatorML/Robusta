import numpy as np
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing

from sklearn.model_selection import check_cv
from category_encoders import *




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

    def __init__(self, smoothing=1.0, min_samples_leaf=1):
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing


    def fit(self, X, y):

        # Category Columns
        cats = X.columns[X.dtypes.astype('str').isin(['object', 'category'])]

        # Fit Smoothed Likelihood Encoder to independent column
        encoder = lambda x: smoothed_likelihood(x, y,
                                                self.min_samples_leaf,
                                                self.smoothing)

        # Save pre-fitted Encoders (key: column name)
        encoders = {col: encoder(X[col].astype('str')) for col in cats}
        self.mapper = lambda x: x.astype('str').map(encoders[x.name]) if x.name in cats else x

        return self


    def transform(self, X):
        # Apply Encoders to each column
        return X.apply(self.mapper)



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

    def __init__(self, smooth=5.0):
        self.smooth = smooth


    def transform(self, X):
        return X.multiply(self._r)


    def fit(self, X, y):
        self._r = np.log(self._pr(X, y, 1) / self._pr(X, y, 0))
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
