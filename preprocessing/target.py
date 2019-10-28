import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target
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



class EncoderCV(BaseEstimator, TransformerMixin):
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
    def __init__(self, encoder, cv=5, n_jobs=None):
        self.encoder = encoder
        self.cv = cv
        self.n_jobs = n_jobs


    def fit(self, X, y, groups=None):

        self.cv_ = self._check_cv(y)

        self.encoders_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit)(clone(self.encoder), X, y, trn)
            for trn, oof in self.cv_.split(X, y, groups))

        return self


    def fit_transform(self, X, y, groups=None):

        self.cv_ = self._check_cv(y)

        preds = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_transform)(clone(self.encoder), X, y, trn, oof)
            for trn, oof in self.cv_.split(X, y, groups))

        return self._mean_preds(preds)[X.columns]


    def transform(self, X):

        preds = Parallel(n_jobs=self.n_jobs)(
            delayed(self._transform)(encoder, X)
            for encoder in self.encoders_)

        return self._mean_preds(preds)[X.columns]


    def _mean_preds(self, preds):
        return pd.concat(preds, axis=1).groupby(level=0, axis=1).mean()


    def _fit(self, encoder, X, y, trn):
        return encoder.fit(X.iloc[trn], y.iloc[trn])


    def _fit_transform(self, encoder, X, y, trn, oof):
        return encoder.fit(X.iloc[trn], y.iloc[trn]).transform(X.iloc[oof])


    def _transform(self, encoder, X):
        return encoder.transform(X)


    def _check_cv(self, y):

        task_type = type_of_target(y)

        if task_type == 'binary':
            classifier = True
        elif task_type == 'continuous':
            classifier = False
        else:
            raise ValueError("Unsupported task type '{}'".format(task_type))

        return check_cv(self.cv, y, classifier)



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



def smoothed_likelihood(x, y, smoothing, min_samples_leaf=1):
    prior = y.mean()
    stats = y.groupby(x).agg(['count', 'mean'])
    smoove = 1 / (1 + np.exp(-(stats['count'] - min_samples_leaf) / smoothing))
    likelihood = prior * (1 - smoove) + stats['mean'] * smoothing
    likelihood[stats['count'] == 1] = prior
    return likelihood
