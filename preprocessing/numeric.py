import pandas as pd
import numpy as np
import scipy

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer

from .base import PandasTransformer
from sklearn import preprocessing




INT_DTYPES = ['Int64', 'Int32', 'Int16', 'Int8', 'UInt32', 'UInt16', 'UInt8']
FLOAT_DTYPES = ['float64', 'float32', 'float16']


class DowncastTransformer(BaseEstimator, TransformerMixin):
    """Downcast numeric columns to the smallest numerical dtype possible
    according to the following rules:

        - ‘integer’ or ‘signed’: smallest signed int dtype (min.: np.int8)
        - ‘unsigned’: smallest unsigned int dtype (min.: np.uint8)
        - ‘float’: smallest float dtype (min.: np.float32)


    Parameters
    ----------
    errors : {‘ignore’, ‘raise’}, default ‘raise’
        If ‘raise’, then non-numeric columns will raise an exception
        If ‘ignore’, then non-numeric columns will be passed with no changes

    copy : bool, default True
        If False, change original dataframe

    """
    def __init__(self, errors='raise', copy=True):
        self.errors = errors
        self.copy = copy


    def fit(self, X, y=None):
        '''Only checks <errors> & <copy> parameters

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform

        Returns
        -------
        self

        '''

        # Select numeric
        self.cols = list(X.columns)
        self.nums = list(X.select_dtypes(np.number))
        self.dtypes = X.dtypes.copy()

        # Check <errors> value
        errors_vals = ['raise', 'ignore']

        if self.errors not in errors_vals:
            raise ValueError('<errors> must be in {}'.format(errors_vals))

        if len(self.nums) < len(self.cols) and self.errors is 'raise':
            cols_diff = list(set(self.cols) - set(self.nums))
            raise ValueError("Found non-numeric columns {}".format(cols_diff))

        return self


    def transform(self, X):
        """Downcast each column to the efficient dtype

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform


        Returns
        -------
        Xt : DataFrame, shape [n_samples, n_features]
            Transformed input.

        """

        # Check columns
        cols_diff = set(self.cols) ^ set(X.columns)
        nums_diff = set(self.nums) ^ set(X.select_dtypes(np.number))

        if len(cols_diff) > 0:
            raise ValueError("Found new columns {}".format(cols_diff))

        if len(nums_diff) > 0:
            raise ValueError("Found new numeric columns {}".format(nums_diff))

        # Fit & transform
        for col, x in X[self.nums].items():
            col_type = self._fit_downcast(x)
            self.dtypes[col] = col_type

        return X.astype(self.dtypes, errors=self.errors, copy=self.copy)


    def _fit_downcast(self, x):

        x_min = x.min()
        x_max = x.max()

        try:
            x = x.astype('Int64')

            col_type = 'Int64'
            col_bits = np.iinfo(col_type).bits

            for int_type in INT_DTYPES:
                int_info = np.iinfo(int_type)

                if (x_min >= int_info.min) \
                and (x_max <= int_info.max) \
                and (col_bits >= int_info.bits):

                    col_bits = int_info.bits
                    col_type = int_type

        except:
            col_type = 'float64'
            col_bits = np.finfo(col_type).bits

            for float_type in FLOAT_DTYPES:
                float_info = np.finfo(float_type)

                if (x_min >= float_info.min) \
                and (x_max <= float_info.max) \
                and (col_bits > float_info.bits):

                    col_bits = float_info.bits
                    col_type = float_type

        return col_type




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

        Xt = RankTransformer(pct=True).fit_transform(X) - 0.5
        Xt = MaxAbsScaler().fit_transform(Xt) * (1 - self.eps)
        Xt = scipy.special.erfinv(Xt)

        return Xt




class RankTransformer(BaseEstimator, TransformerMixin):
    '''Compute numerical data ranks (1 through n) along axis. Equal values are
    assigned a rank that is the average of the ranks of those values.

    Parameters
    ----------
    copy : boolean, optional, default True
        Set to False to perform inplace row normalization and avoid a copy.

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
        self.transformer_ = QuantileTransformer(len(X), **self.params).fit(X)
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
        Xt.loc[:,:] = self.transformer_.transform(X)
        return Xt



class SyntheticFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, pair_sum=True, pair_dif=True, pair_mul=True, pair_div=True, join_X=True, eps=1e-2):
        self.pair_sum = pair_sum
        self.pair_dif = pair_dif
        self.pair_mul = pair_mul
        self.pair_div = pair_div
        self.join_X = join_X
        self.eps = eps


    def fit(self, X, y=None):
        if isinstance(X, pd.core.frame.DataFrame):
            self.columns = X.columns
        else:
            self.columns = ['x_{}'.format(i) for i in range(X.shape[1])]
        return self


    def transform(self, X):

        if isinstance(X, pd.core.frame.DataFrame):
            inds = X.index
        else:
            inds = np.arange(X.shape[0])
            X = pd.DataFrame(X, columns=self.columns, index=inds)


        Xt = pd.DataFrame(index=inds)

        cols_pairs = np.array(list(combinations(self.columns, 2)))
        cols_A = cols_pairs[:,0]
        cols_B = cols_pairs[:,1]

        if self.pair_sum:
            cols = ['{}+{}'.format(a, b) for a, b in cols_pairs]
            F = np.vstack([X[a].values + X[b].values for a, b in cols_pairs]).T
            F = pd.DataFrame(F, index=inds, columns=cols)
            Xt = Xt.join(F)

        if self.pair_dif:
            cols = ['{}-{}'.format(a, b) for a, b in cols_pairs]
            F = np.vstack([X[a].values - X[b].values for a, b in cols_pairs]).T
            F = pd.DataFrame(F, index=inds, columns=cols)
            Xt = Xt.join(F)

        if self.pair_mul:
            cols = ['{}*{}'.format(a, b) for a, b in cols_pairs]
            F = np.vstack([X[a].values * X[b].values for a, b in cols_pairs]).T
            F = pd.DataFrame(F, index=inds, columns=cols)
            Xt = Xt.join(F)

        if self.pair_div:
            cols = ['{}/{}'.format(a, b) for a, b in cols_pairs]
            F = np.vstack([X[a].values / (X[b].values + self.eps) for a, b in cols_pairs]).T
            F = pd.DataFrame(F, index=inds, columns=cols)
            Xt = Xt.join(F)

            cols = ['{}/{}'.format(a, b) for b, a in cols_pairs]
            F = np.vstack([X[a].values / (X[b].values + self.eps) for b, a in cols_pairs]).T
            F = pd.DataFrame(F, index=inds, columns=cols)
            Xt = Xt.join(F)

        if self.join_X:
            Xt = X.join(Xt)

        return Xt




MaxAbsScaler = lambda **params: PandasTransformer(preprocessing.MaxAbsScaler(), **params)
