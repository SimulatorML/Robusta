import pandas as pd
import numpy as np
import scipy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing

from .base import PandasTransformer


__all__ = [
    'GaussRank',
    'RankTransform',
    'MaxAbsScaler',
]




class GaussRank(BaseEstimator, TransformerMixin):
    """Normalize numerical features by Gauss Rank scheme.
    http://fastml.com/preparing-continuous-features-for-neural-networks-with-rankgauss/

    Input normalization for gradient-based models such as neural nets is critical.
    For lightgbm/xgb it does not matter. The best what I found during the past and
    works straight of the box is “RankGauss”. Its based on rank transformation.

    1. First step is to assign a linspace to the sorted features from -1 to 1,
    2. then apply the inverse of error function ErfInv to shape them like gaussians,
    3. then substract the mean. Binary features are not touched with this trafo.

    This works usually much better than standard mean/std scaler or min/max.

    Parameters
    ----------
    eps : float, default=1e-9
        Inversed Error Function (ErfInv) is undefined for x=-1 or x=+1, so
        its argument is clipped to range [-1 + eps, +1 - eps]

    copy : boolean, optional, default True
        Set to False to perform inplace row normalization and avoid a copy.

    """
    def __init__(self, eps=1e-9, copy=True):
        self.eps = eps
        self.copy = copy


    def fit(self, X, y=None):
        '''Does nothing.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        self

        '''
        return self


    def transform(self, X):
        """Transform X using Gauss Rank normalization.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        Xt : DataFrame, shape [n_samples, n_features]
            Normalized input.

        """
        Xt = X.copy() if self.copy else X

        Xt = RankTransform(pct=True).fit_transform(X) - 0.5
        Xt = MaxAbsScaler().fit_transform(Xt) * (1 - self.eps)
        Xt = scipy.special.erfinv(Xt)

        return Xt




class RankTransform(BaseEstimator, TransformerMixin):
    '''Compute numerical data ranks (1 through n) along axis. Equal values are
    assigned a rank that is the average of the ranks of those values.

    Parameters
    ----------
    copy : boolean, optional, default True
        Set to False to perform inplace row normalization and avoid a copy.

    method : {'average', 'min', 'max', 'first', 'dense'}

        - 'average': average rank of group
        - 'min': lowest rank in group
        - 'max': highest rank in group
        - 'first': ranks assigned in order they appear in the array
        - 'dense': like 'min', but rank always increases by 1 between groups

    numeric_only : boolean, default None
        Include only float, int, boolean data. Valid only for DataFrame or Panel

    na_option : {'keep', 'top', 'bottom'}

        - 'keep': leave NA values where they are
        - 'top': smallest rank if ascending
        - 'bottom': smallest rank if descending

    ascending : boolean, default True
        False for ranks by high (1) to low (N)

    pct : boolean, default False
        Computes percentage rank of data

    '''
    def __init__(self, copy=True, **params):
        self.copy = copy
        self.params = params


    def fit(self, X, y=None):
        '''Does nothing.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        self

        '''
        return self


    def transform(self, X):
        """Transform X using rank transformer.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        Xt : DataFrame, shape [n_samples, n_features]
            Transformed input.

        """
        Xt = X.copy() if self.copy else X
        return Xt.rank(axis=0, **self.params)




MaxAbsScaler = lambda **params: PandasTransformer(preprocessing.MaxAbsScaler(), **params)
