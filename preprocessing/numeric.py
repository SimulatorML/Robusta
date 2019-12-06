import pandas as pd
import numpy as np
import scipy

from itertools import combinations

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.preprocessing import Normalizer, normalize
from dask_ml.preprocessing import PolynomialFeatures

from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array

from scipy.special import boxcox


__all__ = [
    'DowncastTransformer',
    'GaussRankTransformer',
    'QuantileTransformer',
    'StandardScaler',
    'RobustScaler',
    'MinMaxScaler',
    'MaxAbsScaler',
    'Normalizer',
    'Winsorizer',
    'SyntheticFeatures',
    'KBinsDiscretizer1D',
    'KBinsDiscretizer',
    'PowerTransformer',
    'Binarizer',
    'PolynomialFeatures',
]




NP_INT_DTYPES = ['int64', 'int32', 'int16', 'int8', 'uint32', 'uint16', 'uint8']
PD_INT_DTYPES = ['Int64', 'Int32', 'Int16', 'Int8', 'UInt32', 'UInt16', 'UInt8']
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
    def __init__(self, numpy_only=True, errors='raise', copy=True):
        self.numpy_only = numpy_only
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
            INT_DTYPES = NP_INT_DTYPES if self.numpy_only else PD_INT_DTYPES

            if (x.astype(INT_DTYPES[0]) != x).any():
                raise ValueError()

            col_type = INT_DTYPES[0]
            col_bits = np.iinfo(col_type).bits

            for int_type in INT_DTYPES:
                int_info = np.iinfo(int_type)

                if (x_min >= int_info.min) \
                and (x_max <= int_info.max) \
                and (col_bits >= int_info.bits):

                    col_bits = int_info.bits
                    col_type = int_type

        except:
            col_type = FLOAT_DTYPES[0]
            col_bits = np.finfo(col_type).bits

            for float_type in FLOAT_DTYPES:
                float_info = np.finfo(float_type)

                if (x_min >= float_info.min) \
                and (x_max <= float_info.max) \
                and (col_bits > float_info.bits):

                    col_bits = float_info.bits
                    col_type = float_type

        return col_type



class QuantileTransformer(QuantileTransformer):

    def transform(self, X):

        return_df = hasattr(X, 'columns')

        if return_df:
            columns = X.columns
            index = X.index

        X = self._check_inputs(X)
        self._check_is_fitted(X)

        self._transform(X, inverse=False)

        if return_df:
            return pd.DataFrame(X, columns=columns, index=index)
        else:
            return X



class GaussRankTransformer(BaseEstimator, TransformerMixin):
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
    ranker : object
        Fitted transformer

    eps : float, default=1e-9
        Inversed Error Function (ErfInv) is undefined for x=-1 or x=+1, so
        its argument is clipped to range [-1 + eps, +1 - eps]

    copy : boolean, optional, default True
        Set to False to perform inplace row normalization and avoid a copy.

    """
    def __init__(self, ranker=QuantileTransformer(), copy=True, eps=1e-9):
        self.ranker = ranker
        self.copy = copy
        self.eps = eps


    def fit(self, X, y=None):
        return self


    def transform(self, X):

        if self.copy:
            X, self.ranker_ = X.copy(), clone(self.ranker)
        else:
            X, self.ranker_ = X, self.ranker

        X = self.ranker_.fit_transform(X)
        X -= 0.5
        X *= 2.0 - self.eps
        X = scipy.special.erfinv(X)

        return X



class Winsorizer(BaseEstimator, TransformerMixin):
    """Winsorization

    Replace extreme values with defined quantiles (0.05 and 0.95 by default).

    Parameters
    ----------
    q_min : float [0..1], default=0.05
        Lower quantile

    q_max : float [0..1], default=0.95
        Upper quantile

    """
    def __init__(self, q_min=0.05, q_max=0.95):
        self.q_min = q_min
        self.q_max = q_max

    def fit(self, X, y=None):

        assert isinstance(self.q_min, float), '<q_min> must be float'
        assert isinstance(self.q_max, float), '<q_max> must be float'
        assert self.q_min < self.q_max, '<q_min> must be smaller than <q_max>'
        assert 0 <= self.q_min <= 1, '<q_min> must be in [0..1]'
        assert 0 <= self.q_max <= 1, '<q_max> must be in [0..1]'

        self.min_ = X.quantile(self.q_min)
        self.max_ = X.quantile(self.q_max)
        return self

    def transform(self, X):
        return X.clip(self.min_, self.max_, axis=1)



class SyntheticFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, pair_sum=True, pair_dif=True, pair_mul=True, pair_div=True,
                 join_X=True, eps=1e-2):
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



class RobustScaler(BaseEstimator, TransformerMixin):
    """Scale features using statistics that are robust to outliers.

    Parameters
    ----------
    with_median : bool, default=True
        If True, center the data before scaling.

    with_scaling : bool, default=True
        If True, scale the data to interquartile range.

    quantiles : tuple (q_min, q_max), 0.0 < q_min < q_max < 1.0
        Default: (0.25, 0.75) = (1st quantile, 3rd quantile) = IQR
        Quantile range used to calculate ``scale_``.

    copy : boolean, optional, default is True
        If False, try to avoid a copy and do inplace scaling instead.

    Attributes
    ----------
    center_ : array of floats
        The median value for each feature in the training set.

    scale_ : array of floats
        The (scaled) interquartile range for each feature in the training set.

    """
    def __init__(self, centering=True, scaling=True, quantiles=(0.25, 0.75),
                 copy=True, eps=1e-3):

        self.centering = centering
        self.scaling = scaling
        self.quantiles = quantiles
        self.copy = copy
        self.eps = eps


    def fit(self, X, y=None):

        q_min, q_max = self.quantiles
        if not 0 <= q_min <= q_max <= 1:
            raise ValueError(f"Invalid quantiles: {self.quantiles}")

        if self.centering:
            self.center_ = X.median()

        if self.scaling:
            self.scale_ = X.quantile(q_max) - X.quantile(q_min)
            self.scale_[self.scale_ < self.eps] = 1

        return self


    def transform(self, X):

        X = X.copy() if self.copy else X

        if self.centering:
            X -= self.center_

        if self.scaling:
            X /= self.scale_

        return X



class StandardScaler(BaseEstimator, TransformerMixin):
    """Standardize features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    with_mean : boolean, default=True
        If True, center the data before scaling.

    with_std : bool, default=True
        If True, scale the data to unit variance.

    copy : boolean, optional, default is True
        If False, try to avoid a copy and do inplace scaling instead.

    Attributes
    ----------
    mean_ : Series of floats
        Mean values

    std_ : Series of floats
        Std values

    """
    def __init__(self, with_mean=True, with_std=True, copy=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy


    def fit(self, X, y=None):

        self.mean_ = X.mean() if self.with_mean else None
        self.std_ = X.std() if self.with_std else None

        return self


    def transform(self, X):

        X = X.copy() if self.copy else X

        if self.with_mean:
            X -= self.mean_

        if self.with_std:
            X /= self.std_

        return X



class MinMaxScaler(BaseEstimator, TransformerMixin):
    """Scale features to [0, 1]

    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.

    Attributes
    ----------
    min_ : Series of floats
        Minimum values

    max_ : Series of floats
        Maximum values

    std_ : Series of floats
        Difference between <max_> and <min_>

    """
    def __init__(self, copy=True):
        self.copy = copy


    def fit(self, X, y=None):

        self.min_ = X.min()
        self.max_ = X.max()
        self.std_ = self.max_ - self.min_

        return self


    def transform(self, X):

        X = X.copy() if self.copy else X

        X -= self.min_
        X /= self.std_

        return X



class MaxAbsScaler(BaseEstimator, TransformerMixin):
    """Scale each feature by its maximum absolute value.

    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.

    Attributes
    ----------
    scale_ : Series of floats
        Mudule of maxabs values

    """
    def __init__(self, copy=True):
        self.copy = copy


    def fit(self, X, y=None):

        a = X.min()
        b = X.max()
        self.scale_ = pd.concat([a, b], axis=1).abs().max(axis=1)

        return self


    def transform(self, X):

        X = X.copy() if self.copy else X
        X /= self.scale_

        return X



class Normalizer(Normalizer):

    def transform(self, X):

        return_df = hasattr(X, 'columns')

        if return_df:
            columns = X.columns
            index = X.index

        X = check_array(X, accept_sparse='csr')
        X = normalize(X, axis=1, copy=self.copy, norm=self.norm)

        if return_df:
            return pd.DataFrame(X, columns=columns, index=index)
        else:
            return X



class KBinsDiscretizer1D(BaseEstimator, TransformerMixin):
    """Continuous feature binning

    Parameters
    ----------
    bins : int or array-like
        Number of bins, or quantiles (if strategy='quantile'), or bin edges
        (if strategy='uniform').

    strategy : {'quantile', 'uniform'}
        If <bins> is int, determines binning type:
            - 'quantile': split feature to the equal size bins
            - 'uniform': split feature by the equal distance edges

        If <bins> is array-like, interpreted as type of passed edges:
            - 'quantile': quantiles (must be monotonic and in range [0..1])
            - 'uniform': exact edges values

    Attributes
    ----------
    bins_ : array of floats
        Defined bins edges

    """
    def __init__(self, bins=5, strategy='quantile'):
        self.bins = bins
        self.strategy = strategy


    def fit(self, y):

        if self.strategy is 'quantile':
            _, self.bins_ = pd.qcut(y, self.bins, retbins=True, duplicates='drop')

        elif self.strategy is 'uniform':
            _, self.bins_ = pd.cut(y, self.bins, retbins=True, duplicates='drop')

        else:
            raise ValueError(f"Unknown strategy value passed: {self.strategy}")

        return self


    def transform(self, y):
        return pd.cut(y, self.bins_)



class KBinsDiscretizer(KBinsDiscretizer1D):
    """Continuous feature binning

    Parameters
    ----------
    bins : int or array-like
        Number of bins, or quantiles (if strategy='quantile'), or bin edges
        (if strategy='uniform').

    strategy : {'quantile', 'uniform'}
        If <bins> is int, determines binning type:
            - 'quantile': split feature to the equal size bins
            - 'uniform': split feature by the equal distance edges

        If <bins> is array-like, interpreted as type of passed edges:
            - 'quantile': quantiles (must be monotonic and in range [0..1])
            - 'uniform': exact edges values

    Attributes
    ----------
    bins_ : dict of array of floats
        Defined bins edges

    """
    def fit(self, X, y=None):

        self.transformers = {}
        self.bins_ = {}

        for col in X:
            params = self.get_params()
            transformer = KBinsDiscretizer1D(**params).fit(X[col])
            self.transformers[col] = transformer
            self.bins_[col] = transformer.bins_

        return self


    def transform(self, X):
        Xt = pd.DataFrame(index=X.index)

        for col, transformer in self.transformers.items():
            Xt[col] = transformer.transform(X[col])

        return Xt



class PowerTransformer(PowerTransformer):

    def fit_transform(self, X, y=None):

        return_df = hasattr(X, 'columns')
        if return_df:
            columns = X.columns
            index = X.index

        X = self._fit(X, y, force_transform=True)

        if return_df:
            return pd.DataFrame(X, columns=columns, index=index)
        else:
            return X

    def transform(self, X):

        return_df = hasattr(X, 'columns')
        if return_df:
            columns = X.columns
            index = X.index

        check_is_fitted(self, 'lambdas_')
        X = self._check_input(X, check_positive=True, check_shape=True)

        transform_function = {'box-cox': boxcox,
                              'yeo-johnson': self._yeo_johnson_transform
                              }[self.method]
        for i, lmbda in enumerate(self.lambdas_):
            with np.errstate(invalid='ignore'):  # hide NaN warnings
                X[:, i] = transform_function(X[:, i], lmbda)

        if self.standardize:
            X = self._scaler.transform(X)

        if return_df:
            return pd.DataFrame(X, columns=columns, index=index)
        else:
            return X

        return X



class Binarizer(BaseEstimator, TransformerMixin):
    '''Binarize data (set feature values to 0 or 1) according to a threshold

    Values greater than the threshold map to 1, while values less than
    or equal to the threshold map to 0. With the default threshold of 0,
    only positive values map to 1.

    Parameters
    ----------
    threshold : float or array-like, shape (n_features, )
        Single or array of threshold values

    '''
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X > self.threshold).astype('uint8')



class PolynomialFeatures(PolynomialFeatures):

    def __init__(self, degree=2, interaction_only=False, include_bias=True,
                 preserve_dataframe=True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.preserve_dataframe = preserve_dataframe
