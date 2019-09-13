import pandas as pd
import numpy as np

import time
import abc

from sklearn.base import TransformerMixin

from robusta.crossval import crossval_score

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
        use_cols = self._select_features()

        Xt = X[use_cols]
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

    def _eval_subset(self, subset, X, y, groups):

        self.n_features_ = len(X.columns)

        time_start = time.time()

        scores = crossval_score(self.estimator, self.cv, X[subset], y, groups,
                                scoring=self.scoring, n_jobs=self.n_jobs,
                                verbose=0)

        # TODO: multimetric case (or check if multimetric)
        score = scores[self.scoring].mean()

        time_end = time.time()

        trial = {
            'subset': subset,
            'score': score,
            'time': time_end-time_start,
        }

        self._append_trial(trial)

        return score



    def _append_trial(self, trial):

        if not hasattr(self, 'trials_'):
            self.trials_ = pd.DataFrame()
            self.best_score_ = -np.inf

            self.total_time_ = .0
            self.n_trials_ = 0

        if trial['score'] > self.best_score_:
            self.best_trial_ = self.n_trials_ - 1
            self.best_score_ = trial['score']
            self.best_subset_ = trial['subset']

        self.trials_ = self.trials_.append(trial, ignore_index=True)

        self.time_ = self.trials_['time'].sum()
        self.n_trials_ = self.trials_.shape[0]

        _print_last(self)

        self._check_max_trials()
        self._check_max_time()


    def _check_max_trials(self):
        if self.max_trials and self.n_trials_ >= self.max_trials:
            if self.verbose: print('Iterations limit exceed!')
            raise KeyboardInterrupt


    def _check_max_time(self):
        if self.max_time and self.time_ >= self.max_time:
            if self.verbose: print('Time limit exceed!')
            raise KeyboardInterrupt
