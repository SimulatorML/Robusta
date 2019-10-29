import pandas as pd
import numpy as np
import abc

from time import time

from sklearn.base import TransformerMixin

from robusta.crossval import crossval

from ._verbose import _print_last



class Selector(TransformerMixin):


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
        features = self.get_features()
        return X[features]


    @abc.abstractmethod
    def get_features(self):
        """
        Get list of columns to select

        Returns
        -------
        use_cols : list of string, shape (k_features, )
            Columns to selct

        """
        return []



class BlackBoxSelector(Selector):

    @abc.abstractmethod
    def __init__(self, estimator, cv=5, scoring=None, max_iter=20, max_time=None,
                 random_state=0, n_jobs=-1, verbose=1, n_digits=4, plot=False):

        self.estimator = estimator
        self.scoring = scoring
        self.max_iter = max_iter
        self.max_time = max_time
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_digits = n_digits
        self.plot = plot


    @property
    def n_features_(self):
        return len(self.features_)


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



    def _eval_subset(self, subset, X, y, groups, **kwargs):

        trial = self._find_trial(subset)

        if not trial:
            tic = time()

            features = list(subset)
            result = crossval(self.estimator, self.cv, X[features], y, groups,
                              scoring=self.scoring, n_jobs=self.n_jobs,
                              return_pred=False, verbose=0, **kwargs)

            trial = {
                'score': np.mean(result['score']),
                'score_std': np.std(result['score']),
                'subset': result['features'],
                'time': time() - tic,
            }

            if 'importance' in result:
                trial['importance'] = result['importance'].mean(axis=0)
                trial['importance_std'] = result['importance'].std(axis=0)

        self._append_trial(trial)

        return trial


    def _append_trial(self, trial):

        if not hasattr(self, 'trials_'):
            self._reset_trials()

        self.trials_ = self.trials_.append(trial, ignore_index=True)

        _print_last(self)

        self._check_max_iter()
        self._check_max_time()


    def _reset_trials(self):
        self.trials_ = pd.DataFrame()


    @property
    def total_time_(self):
        return self.trials_['time'].sum() if hasattr(self, 'trials_') else .0


    @property
    def n_iters_(self):
        return self.trials_.shape[0] if hasattr(self, 'trials_') else 0

    @property
    def best_iter_(self):
        return self.trials_['score'].idxmax() if hasattr(self, 'trials_') else None


    @property
    def best_score_(self):
        if hasattr(self, 'trials_'):
            return self.trials_.loc[self.best_iter_, 'score']
        else:
            return -np.inf


    @property
    def best_subset_(self):
        return self.trials_.loc[self.best_iter_, 'subset']


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


    def _find_trial(self, subset):

        if self.n_iters_ == 0:
            return None

        same_subsets = self.trials_['subset'].map(lambda x: _same_set(subset, x))

        if same_subsets.any():
            trial = self.trials_[same_subsets].iloc[0]
            return trial.to_dict()

        else:
            return None


    @property
    def feature_importances_(self):
        subset = self._select_features()
        trial = _find_trial(subset)
        return pd.Series(trial['importance'], index=self.features_)


    @property
    def feature_importances_std_(self):
        subset = self._select_features()
        trial = _find_trial(subset)
        return pd.Series(trial['importance_std'], index=self.features_)


def _same_set(set1, set2):
    set1, set2 = set(set1), set(set2)
    return len(set1 ^ set2) is 0



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
