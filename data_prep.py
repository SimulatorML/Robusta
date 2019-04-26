from pylab import *
import pandas as pd
import numbers

from xam.feature_extraction import *
from sklearn.preprocessing import maxabs_scale, robust_scale, quantile_transform, power_transform
from scipy.special import erfinv

from robusta import utils







'''
Numerical (num)
'''

def num_minmax(X, cols, prefix='01_', **params):
    # MinMaxScaler
    _X = X.copy()
    _cols = [prefix+col for col in cols]
    a, b = X[cols].min(), X[cols].max()
    _X[_cols] = (X[cols] - a)/(b - a)
    return _X, _cols

def num_maxabs(X, cols, prefix='ma_', **params):
    # MaxAbsScaler
    _X = X.copy()
    _cols = [prefix+col for col in cols]
    _x = maxabs_scale(X[cols], **params)
    _X[_cols] = pd.DataFrame(_x, columns=_cols, index=_X.index)
    return _X, _cols

def num_robust(X, cols, prefix='rob_', **params):
    # RobustScaler
    _X = X.copy()
    _cols = [prefix+col for col in cols]
    _x = robust_scale(X[cols], **params)
    _X[_cols] = pd.DataFrame(_x, columns=_cols, index=_X.index)
    return _X, _cols

def num_std(X, cols, prefix='std_', **params):
    # StandardScaler
    _X = X.copy()
    _cols = [prefix+col for col in cols]
    m, s = X[cols].mean(), X[cols].std()
    _X[_cols] = (X[cols] - m)/s
    return _X, _cols

def num_root(X, cols, prefix='root_', **params):
    # SqrtTransform
    _X = X.copy()
    _cols = [prefix+col for col in cols]
    _X[_cols] = _X[cols].applymap(np.sqrt)
    return _X, _cols

def num_log(X, cols, prefix='log_', **params):
    # LogTransform
    _X = X.copy()
    _cols = [prefix+col for col in cols]
    _X[_cols] = _X[cols].applymap(np.log1p)
    return _X, _cols

def num_binning(X, cols, prefix='bin_', **params):
    # Binning
    _X = X.copy()
    _cols = [prefix+col for col in cols]
    if 'bins' not in params:
        params['bins'] = 10
    if 'labels' not in params:
        params['labels'] = False
    _X[_cols] = X[cols].apply(lambda s: pd.cut(s, **params)).fillna(-1)
    _X[_cols] = _X[_cols].astype(int)
    return _X, _cols

def num_cycle(X, cols, suffix=['_cos', '_sin'], **params):
    # CyclicEncoding
    _X = X.copy()
    _cols = [col+suffix[0] for col in cols] + [col+suffix[1] for col in cols]
    encoder = CycleTransformer()
    _X[_cols] = pd.DataFrame(encoder.fit_transform(X[cols]),
                             columns=_cols, index=_X.index)
    return _X, _cols

def num_rank(X, cols, prefix='rank_', **params):
    # RankTransform
    _X = X.copy()
    _cols = [prefix+col for col in cols]
    if 'pct' not in params:
        params['pct'] = True
    _X[_cols] = X[cols].rank(**params)
    return _X, _cols

def num_gaus(X, cols, prefix='gr_', **params):
    # GaussRank Normalization
    if 'eps' not in params:
        params['eps'] = 1e-6
    _X = X.copy()
    _cols = [prefix+col for col in cols]
    _X[_cols] = _X[cols]
    _X = num_rank(_X, _cols, prefix='')[0]
    _X[_cols] -= 0.5
    _X = num_maxabs(_X, _cols, prefix='')[0]
    _X[_cols] *= 1-params['eps']
    _X[_cols] = erfinv(_X[_cols])
    return _X, _cols

def num_quantile(X, cols, predix='q_', **params):
    # QuantileTransformer
    if 'random_state' not in params:
        params['random_state'] = 0
    _X = X.copy()
    _cols = [prefix+col for col in cols]
    _x = quantile_transform(X[cols], **params)
    _X[_cols] = pd.DataFrame(_x, columns=_cols, index=_X.index)
    return _X, _cols

def num_boxcox(X, cols, predix='bc_', **params):
    # Box-Cox Transformer
    params['method'] = 'box-cox'
    _X = X.copy()
    _cols = [prefix+col for col in cols]
    _x = power_transform(X[cols])
    _X[_cols] = pd.DataFrame(_x, columns=_cols, index=_X.index)
    return _X, _cols

def num_yeo(X, cols, predix='yeo_', **params):
    # Yeo-Johnson Transformer
    params['method'] = 'yeo-johnson'
    _X = X.copy()
    _cols = [prefix+col for col in cols]
    _x = power_transform(X[cols])
    _X[_cols] = pd.DataFrame(_x, columns=_cols, index=_X.index)
    return _X, _cols


'''
Categorical (cat)
+ ordinal, binary
'''
def cat_cat(X, cols, prefix='', **params):
    # LabelEncoding
    _X = X.copy()
    _cols = [prefix+col for col in cols]

    for _col, col in zip(_cols, cols):
        _X[_col] = _X[col].astype('category')

    return _X, _cols


def cat_le(X, cols, prefix='le_', **params):
    # LabelEncoding
    _X = X.copy()
    _cols = [prefix+col for col in cols]

    for _col, col in zip(_cols, cols):
        _X[_col] = _X[col].astype('category').cat.codes

    return _X, _cols


def cat_fe(X, cols, prefix='fe_', **params):
    # FreqEncoding
    _X = X.copy()
    _cols = [prefix+col for col in cols]
    _X[_cols] = X[cols].apply(lambda s: s.map(s.value_counts(True)))
    return _X, _cols


def cat_ohe(X, cols, suffix='_ohe_', **params):
    # OneHotEncoding
    _X = X.copy()
    _x = pd.get_dummies(X[cols], prefix_sep=suffix, columns=cols)
    _cols = list(_x.columns)
    _X[_cols] = _x
    return _X, _cols

def cat_comb(X, cols, sep='+', **params):
    # Combinations
    _X = X.copy()
    orders = params['orders'] if 'orders' in params else [2]
    encoder = FeatureCombiner(separator=sep, orders=orders)
    _x = encoder.fit_transform(X[cols].astype('category'))
    _cols = list(_x.columns)
    _X[_cols] = pd.DataFrame(_x, columns=_cols, index=_X.index)
    return _X, _cols



'''
Text (txt)
'''
def txt_chars(X, cols, suffix='_#chars', **params):
    # chars count
    _X = X.copy()
    _cols = [col+suffix for col in cols]
    _X[_cols] = _X[cols].applymap(len)
    return _X, _cols

def txt_words(X, cols, suffix='_#words', **params):
    # words count
    _X = X.copy()
    _cols = [col+suffix for col in cols]
    _X[_cols] = _X[cols].applymap(utils.count_words)
    return _X, _cols

def txt_uwords(X, cols, suffix='_#uwords', **params):
    # words count
    _X = X.copy()
    _cols = [col+suffix for col in cols]
    _X[_cols] = _X[cols].applymap(utils.count_unique_words)
    return _X, _cols

def txt_avgchars(X, cols, suffix='_avg#chars', **params):
    # avg chars per word
    _X = X.copy()
    _cols = [col+suffix for col in cols]
    C = _X[cols].applymap(lambda x: len(''.join(utils.get_words(x))))
    W = _X[cols].applymap(utils.count_words)
    _X[_cols] = C / W
    return _X, _cols
