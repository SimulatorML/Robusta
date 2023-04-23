import pandas as pd
import numpy as np
import abc

from copy import copy
from time import time

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.random import check_random_state

from robusta.crossval import crossval

from ._verbose import _print_last
from ._subset import FeatureSubset
from ._plot import _plot_progress, _plot_subset





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



class _WrappedSelector(_Selector):

    @abc.abstractmethod
    def __init__(self, estimator, cv=5, scoring=None, max_iter=20, max_time=None,
                 random_state=0, n_jobs=-1, verbose=1, n_digits=4, cv_kwargs={}):

        self.estimator = estimator
        self.scoring = scoring
        self.max_iter = max_iter
        self.max_time = max_time
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_digits = n_digits
        self.cv_kwargs = cv_kwargs


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



    def _get_importance(self, subset, result):
        if 'importance' in result:
            imp = result['importance']
            subset.importance = pd.Series(np.average(imp, axis=0), index=subset)
            subset.importance_std = pd.Series(np.std(imp, axis=0), index=subset)
        return subset



    def _eval_subset(self, subset, X, y, groups):

        result = crossval(self.estimator, self.cv, X[subset], y, groups,
                          scoring=self.scoring, n_jobs=self.n_jobs,
                          return_pred=False, verbose=0,
                          **self.cv_kwargs)

        subset.score = np.average(result['val_score'])
        subset.score_std = np.std(result['val_score'])
        subset = self._get_importance(subset, result)

        return subset



    def eval_subset(self, subset, X, y, groups=None):

        # Convert to FeatureSubset
        if type(subset) != type(self.features_):
            subset = self.features_.copy().set_subset(subset)

        # Evaluate
        tic = time()
        self._eval_subset(subset, X, y, groups)
        subset.eval_time = time() - tic

        # Update stats
        self.total_time_ = getattr(self, 'total_time_', .0) + subset.eval_time

        if not hasattr(self, 'best_score_') or self.best_score_ < subset.score:
            self.best_subset_ = subset
            self.best_score_  = subset.score

        # Update history
        subset.idx = self.n_iters_
        self.trials_.append(subset)

        # Verbose
        _print_last(self)

        # Check limits
        self._check_max_iter()
        self._check_max_time()

        return subset.score


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


    def plot_progress(self, **kwargs):
        return _plot_progress(self, **kwargs)

    def plot_subset(self, **kwargs):
        return _plot_subset(self, **kwargs)

    def get_subset(self):

        if hasattr(self, 'best_subset_'):
            return self.best_subset_
        else:
            model_name = self.__class__.__name__
            raise NotFittedError(f'{model_name} is not fitted')




def _check_k_features(k_features, n_features, param='k_features'):

    if isinstance(k_features, int):
        if k_features > 0:
            k_features = k_features
        else:
            raise ValueError(f'Integer <{param}> must be greater than 0')

    elif isinstance(k_features, float):
        if 0 < k_features < 1:
            k_features = max(k_features * n_features, 1)
            k_features = int(k_features)
        else:
            raise ValueError(f'Float <{param}> must be from interval (0, 1)')

    else:
        raise ValueError(f'Parameter <{param}> must be int or float,'
                         f'got {k_features}')

    return k_features




class _WrappedGroupSelector:
    def _get_importance(subset,
                        result):
        if 'importance' in result:
            features, imp = result['features'], result['importance']
            groups = [group for group, _ in features]

            imp = pd.DataFrame(imp, columns=groups).T
            imp = imp.groupby(groups).sum()

            subset.importance = imp.mean(axis=1)
            subset.importance_std = imp.std(axis=1)
        return subset

    def _set_features(self, X):
        self.features_ = FeatureSubset(X.columns, group=True)
