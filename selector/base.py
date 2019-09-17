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
        features = self._select_features()
        features = list(features)

        Xt = X[features]
        return Xt


    @abc.abstractmethod
    def _select_features(self):
        """
        Get list of columns to select

        Returns
        -------
        use_cols : list of string, shape (k_features, )
            Columns to selct

        """
        return []



class EmbeddedSelector(Selector):

    @abc.abstractmethod
    def __init__(self, estimator, scoring=None, cv=5, max_iter=20, max_time=None,
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


    def _eval_subset(self, subset, X, y, groups, **kwargs):

        trial = self._find_trial(subset)

        if trial:
            score = trial['score']

        else:
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
                trial['importance'] = result['importance'].mean(axis=1)
                trial['importance_std'] = result['importance'].std(axis=1)

        self._append_trial(trial)

        return trial


    def _append_trial(self, trial):

        if not hasattr(self, 'trials_'):
            self._reset_trials()

        if trial['score'] >= self.best_score_:
            self.best_iter_ = self.n_iters_
            self.best_score_ = trial['score']
            self.best_subset_ = trial['subset']

        self.trials_ = self.trials_.append(trial, ignore_index=True)

        self.total_time_ = self.trials_['time'].sum()
        self.n_iters_ = self.trials_.shape[0]

        _print_last(self)

        self._check_max_trials()
        self._check_max_time()


    def _reset_trials(self):

        self.trials_ = pd.DataFrame()
        self.best_score_ = -np.inf

        self.total_time_ = .0
        self.n_iters_ = 0


    def _check_max_trials(self):
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
