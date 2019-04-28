from pylab import *
import pandas as pd
import numpy as np

from xam.feature_extraction import BayesianTargetEncoder
from sklearn.model_selection import KFold





'''
Mean Target Encoder

    X_train = te.fit_transform(X_train, y_train, cols)
    X_test = te.transform(X_test)
'''

class BaseTE():

    def __init__(self, params={}):
        self.params = params

    def fit_transform(self, X, y, cols=[]):
        self.trans_func = lambda x: x
        return X

    def transform(self, X, y=None):
        return self.trans_func(X)


class SmoothTE(BaseTE):

    def fit_transform(self, X, y, cols=[]):
        self.alpha = self.params['alpha'] if 'alpha' in self.params else 50

        encoder = BayesianTargetEncoder(columns=cols, suffix='',
                                        prior_weight=self.alpha)

        _X = encoder.fit_transform(X.copy(), y)
        self.trans_func = lambda x: encoder.transform(x.copy())
        return _X



class NestedTE(BaseTE):

    def fit_transform(self, X, y, cols=[]):
        self.alpha = self.params['alpha'] if 'alpha' in self.params else 50
        self.folds = self.params['folds'] if 'folds' in self.params else 10

        if isinstance(self.folds, int):
            #try:
            #    folds = list(StratifiedKFold(self.folds, True, 0).split(X, y))
            #except:
            folds = list(KFold(self.folds, True, 0).split(X, y))
        else:
            folds = self.folds

        params = dict(self.params)
        params.update({'alpha': self.alpha})

        _X = X.copy()
        transformers = []
        for trn, val in folds:
            X_trn, X_val = X.iloc[trn], X.iloc[val]
            y_trn, y_val = y.iloc[trn], y.iloc[val]
            te_smooth = SmoothTE(params)
            X_i = te_smooth.fit_transform(X_trn, y_trn, cols)
            T = lambda x: te_smooth.transform(x)
            transformers.append(T)
            _X.iloc[val] = T(X_val)
        _X = _X.astype(float)

        def trans(x):
            n = len(transformers)
            for i, T in enumerate(transformers):
                t = T(x)
                _x = _x.add(t) if i else t
            _x /= n
            return _x

        self.trans_func = trans
        return _X



#def te_expand(X, y, cols, suffix='', **params):
#    # Not finished
#    n_iters = params['n_iters'] if 'n_iters' in params else 10
#
#    _X = X.copy()
#    _X['target'] = y
#
#    for i in range(n_iters):
#        for col in cols:
#            cumcnt = _X.groupby(col)['target'].cumcount()
#            cumsum = _X.groupby(col)['target'].cumsum() - _X['target']
#            _X[col] = cumsum/cumcnt
#
#    #_X = encoder.fit_transform(X.copy(), y)
#    #trans = lambda x: encoder.transform(x.copy())

    #return _X, trans

'''
Target Transformer

    y_train = tt.transform(y_train)
    y_pred = tt.inverse_transform(y_pred)
'''
class BaseTT():
    # y := y
    def __init__(self):
        self.fun = lambda y: y
        self.inv = lambda y: y

    def transform(self, y):
        self.y_type = y.dtype
        return self.fun(y)

    def inverse_transform(self, y):
        return self.inv(y)#.astype(self.y_type)


class CustomTT(BaseTT):
    def __init__(self, func, inverse_func):
        self.fun = lambda y: func(y)
        self.inv = lambda y: inverse_func(y)


class LogTT(BaseTT):
    # y := log(y)
    def __init__(self):
        self.fun = lambda y: y.map(log)
        self.inv = lambda y: y.map(exp)

class Log1TT(BaseTT):
    # y := log(y+1)
    def __init__(self):
        self.fun = lambda y: y.map(log1p)
        self.inv = lambda y: y.map(expm1)

class SqrtTT(BaseTT):
    # y := sqrt(y)
    def __init__(self):
        self.fun = lambda y: y.map(np.sqrt)
        self.inv = lambda y: y.map(np.square)



# TODO: timeseries transformations
# groupby !!!
class DifTT(BaseTT):
    # y(t) := y(t) - y(y-1)
    def __init__(self):
        self.fun = lambda y: y.diff()
        self.inv = lambda y: y.cumsum()

class CumTT(BaseTT):
    # y(t) := ∑[i=0..t, y(i)]
    def __init__(self):
        self.fun = lambda y: y.cumsum()
        self.inv = lambda y: y.diff()


'''
Imbalance Classes Resampling

    X, y = imba.fit_resample(X, y)
'''
from imblearn import under_sampling, over_sampling, combine

class BaseResampler():

    def __init__(self, **params):
        self.params = params
        self.resampler = lambda X, y: (X, y)

    def fit_resample(self, X, y):
        X_re, y_re = self.resampler(X, y)

        X_re = self._X_format(X_re, X)
        y_re = self._y_format(y_re, y)

        return X_re, y_re

    def _X_format(self, _X, X):
        _X = pd.DataFrame(_X, columns=X.columns)
        _X = _X.astype(X.dtypes)
        return _X

    def _y_format(self, _y, y):
        try:
            _y = pd.Series(_y, name=y.name)
        except:
            _y = pd.DataFrame(_y, name=y.columns)
        return _y



'''
Transformers & Encoders dicts
'''
target_transformers = {
    None: BaseTT(),
    'custom': lambda f, g: CustomTT(f, g),
    'log': LogTT(),
    'log1p': Log1TT(),
    'sqrt': SqrtTT()
}

# te_params
target_encoders = {
    None: BaseTE,
    'inbuilt': BaseTE, # for model's inbuilt categorical encoder
    'smooth': SmoothTE,
    'nested': NestedTE,
}

class TE(BaseTE):

    def __init__(self, name, **params):
        te = target_encoders[name](params)
        self.resampler = lambda X, y: resampler.fit_resample(X, y)

# imba_params
resamplers = {
    None: BaseResampler,
    'CC': under_sampling.ClusterCentroids,
    'RUS': under_sampling.RandomUnderSampler,
    'CNN': under_sampling.CondensedNearestNeighbour,
    'ENN': under_sampling.EditedNearestNeighbours,
    'RENN': under_sampling.RepeatedEditedNearestNeighbours,
    'AllKNN': under_sampling.AllKNN,
    'IHT': under_sampling.InstanceHardnessThreshold,
    'NM': under_sampling.NearMiss,
    'NCR': under_sampling.NeighbourhoodCleaningRule,
    'OSS': under_sampling.OneSidedSelection,
    'TL': under_sampling.TomekLinks,
    'ROS': over_sampling.RandomOverSampler,
    'ADASYN': over_sampling.ADASYN,
    'SMOTE': over_sampling.SMOTE,
    'SMOTEENN': combine.SMOTEENN,
    'SMOTET': combine.SMOTETomek
}

class Imba(BaseResampler):

    def __init__(self, name, **params):
        resampler = resamplers[name](**params)
        self.resampler = lambda X, y: resampler.fit_resample(X, y)


def get_TE(name, params):
    return target_encoders[name](params)

def get_TT(name):
    return target_transformers[name]

# from prep_inner import TT, TE, Imba
#
# tt = get_TT(name)
# y = tt.transform(y)
# y_pred = tt.inverse_transform(y_pred)
#
# te = get_TE(name, params)
# X = te.fit_transform(X, y, cols)
# X_test = te.transform(X_test)
#
# imba = Imba(name, params)
# X, y = imba.fit_resample(X, y)
