import pandas as pd
import numpy as np
import abc

from copy import copy
from time import time

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.random import check_random_state

from robusta.crossval import crossval

from ._verbose import _print_last
from ._plot import _plot_progress




class FeatureSubset:

    def __init__(self, features, subset=None, mask=None, group=False):

        # Features
        msg = '<features> must be unique'
        assert len(set(features)) == len(features), msg

        if group:
            self.features = features.get_level_values(0).unique()
        else:
            self.features = np.array(features)

        # subset OR mask
        if subset is not None and mask is not None:
            raise ValueError('<subset> & <mask> could not be set at once')

        elif subset is not None:
            self.set_subset(subset)

        elif mask is not None:
            self.set_mask(mask)

        else:
            self.set_mask([True]*self.n_features)


    def __iter__(self):
        return iter(self.subset)


    def __len__(self):
        return self.n_selected


    def __array__(self, *args, **kwargs):
        return np.array(self.subset, *args, **kwargs)


    def __str__(self):
        return self.subset.__str__()


    def __repr__(self):
        nm = self.__class__.__name__
        st = self.__str__().replace('\n', '\n ' + ' '*len(nm))
        return '{}({})'.format(nm, st)


    def __eq__(self, other):
        return np.all(self.mask == other.mask)


    def set_subset(self, subset):

        msg = 'Not all <subset> values are in <features>'
        assert np.isin(subset, self.features).all(), msg

        msg = 'All <subset> values must be unique'
        assert len(set(subset)) == len(subset), msg

        self.set_mask(np.isin(self.features, subset))

        return self


    def set_mask(self, mask):

        msg = '<mask> length must be the same as <features>'
        assert len(mask) == self.n_features, msg

        self.mask = np.array(mask, dtype=bool)
        self.subset = self.features[self.mask]

        return self


    def sample(self, size=None, random_state=None):

        rstate = check_random_state(random_state)

        if size:
            subset = rstate.choice(self.features, size=size, replace=False)
            return self.copy().set_subset(subset)

        else:
            mask = rstate.randint(0, 2, size=self.n_features, dtype=bool)
            return self.copy().set_mask(mask)


    def remove(self, *features, copy=True):

        self = self.copy() if copy else self

        msg = 'All elements must be unique'
        assert len(set(features)) == len(features), msg

        msg = 'All elements must be in <subset>'
        assert np.isin(features, self.subset).all(), msg

        mask = np.isin(self.features, features)
        self.set_mask(self.mask ^ mask)

        return self


    def append(self, *features, copy=True):

        self = self.copy() if copy else self

        msg = 'All elements must be unique'
        assert len(set(features)) == len(features), msg

        msg = 'All elements must be in <features>'
        assert np.isin(features, self.features).all(), msg

        msg = 'Some elements already in <subset>'
        assert not np.isin(features, self.subset).any(), msg

        self.set_subset(np.append(self.subset, features))

        return self


    def copy(self):
        return copy(self)

    @property
    def n_features(self):
        return len(self.features)

    @property
    def n_selected(self):
        return len(self.subset)

    @property
    def shape(self):
        return (self.n_selected, )




class _Selector(BaseEstimator, TransformerMixin):


    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : DataFrame of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        Xt : DataFrame of shape [n_samples, n_selected_features]
            The input samples with only the selected features.

        """
        return X[self.get_subset()]


    @abc.abstractmethod
    def get_subset(self):
        """
        Get list of columns to select

        Returns
        -------
        use_cols : list of string, shape (k_features, )
            Columns to selct

        """
        return self.features_


    def _set_features(self, X):
        self.features_ = FeatureSubset(X.columns)




class _GroupSelector:

    def _set_features(self, X):
        self.features_ = FeatureSubset(X.columns, group=True)



class _AgnosticSelector(_Selector):

    @abc.abstractmethod
    def __init__(self, estimator, cv=5, scoring=None, max_iter=20, max_time=None,
                 random_state=0, n_jobs=-1, verbose=1, n_digits=4):

        self.estimator = estimator
        self.scoring = scoring
        self.max_iter = max_iter
        self.max_time = max_time
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_digits = n_digits


    @property
    def n_features_(self):
        return self.features_.n_features


    @property
    def min_features_(self):
        min_features = _check_k_features(self.min_features,
                                         self.n_features_,
                                         'min_features')
        return min_features


    @property
    def max_features_(self):
        max_features = _check_k_features(self.max_features,
                                         self.n_features_,
                                         'max_features')
        msg = "<min_features> must be lower then <max_features>"
        assert self.min_features_ <= max_features, msg
        return max_features



    def _eval_subset(self, subset, X, y, groups=None):

        tic = time()

        # Convert to FeatureSubset
        if type(subset) != type(self.features_):
            subset = self.features_.copy().set_subset(subset)

        # Evaluate subset
        result = crossval(self.estimator, self.cv, X[subset], y, groups,
                          scoring=self.scoring, n_jobs=self.n_jobs,
                          return_pred=False, verbose=0)

        subset.time  = time() - tic
        subset.score     = np.mean(result['score'])
        subset.score_std = np.std(result['score'])

        # Extract importances
        if 'importance' in result and len(subset) == len(result['features']):
            imp = result['importance']
            subset.importance = pd.Series(np.mean(imp, axis=0), index=subset)

        # Update history
        subset.idx = self.n_iters_
        self.trials_.append(subset)

        # Update stats
        self.total_time_ = getattr(self, 'total_time_', .0) + subset.time

        if not hasattr(self, 'best_score_') or self.best_score_ < subset.score:
            self.best_subset_ = subset
            self.best_score_  = subset.score

        # Verbose
        _print_last(self)

        # Check limits
        self._check_max_iter()
        self._check_max_time()


    def _check_max_iter(self):
        if hasattr(self, 'max_iter') and self.max_iter:
            if self.max_iter <= self.n_iters_:
                if self.verbose: print('Iterations limit exceed!')
                raise KeyboardInterrupt


    def _check_max_time(self):
        if hasattr(self, 'max_time') and self.max_time:
            if self.max_time <= self.total_time_:
                if self.verbose: print('Time limit exceed!')
                raise KeyboardInterrupt


    def _reset_trials(self):
        self.trials_ = []


    @property
    def n_iters_(self):
        return len(self.trials_)


    #@property
    #def feature_importances_(self):
    #    subset = self._select_features()
    #    trial = _find_trial(subset)
    #    return pd.Series(trial['importance'], index=self.features_)


    #@property
    #def feature_importances_std_(self):
    #    subset = self._select_features()
    #    trial = _find_trial(subset)
    #    return pd.Series(trial['importance_std'], index=self.features_)


    def plot(self, **kwargs):
        _plot_progress(self, **kwargs)




def _check_k_features(k_features, n_features, param='k_features'):

    if isinstance(k_features, int):
        if k_features > 0:
            k_features = k_features
        else:
            raise ValueError('Integer <{}> must be greater than 0'
                             ''.format(param))

    elif isinstance(k_features, float):
        if 0 < k_features < 1:
            k_features = max(k_features * n_features, 1)
            k_features = int(k_features)
        else:
            raise ValueError('Float <{}> must be from interval (0, 1)'
                             ''.format(param))

    else:
        raise ValueError('Parameter <{}> must be int or float, '
                         'got {}'.format(param, k_features))

    return k_features
