import pandas as pd
import numpy as np

import abc

from sklearn.base import TransformerMixin

from robusta.crossval import crossval_score




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

        time_start = time.time()

        scores = crossval_score(estimator, self.cv, X[subset], y, groups,
                                scoring=self.scoring, n_jobs=self.n_jobs)
        score = np.mean(scores)

        time_end = time.time()

        self._append_trial(subset, score=score, time=time_end-time_start)

        return score


    def _append_trial(self, subset, score, time):

        if not hasattr(self, 'trials_'):
            self.trials_ = pd.DataFrame(columns=['subset', 'score', 'time'])
            self.best_score_ = -np.inf

            self.total_time_ = .0
            self.n_trials_ = 0

        if score > self.best_score_:
            self.best_trial_ = len(self.trials_) - 1
            self.best_score_ = score
            self.best_subset_ = subset

        self.trials_.loc[self.n_trials_] = [subset, score, time]

        self.time_ = self.trials_['time'].sum()
        self.n_trials_ = self.trials_.shape[0]

        self._check_max_trials()
        self._check_max_time()


    def _check_max_trials(self):

        if self.max_trials and self.max_trials <= self.n_trials_:
            if self.verbose:
                print('Iterations limit exceed')
            raise KeyboardInterrupt


    def _check_max_time(self):

        if self.max_time and self.max_time <= self.time_:
            if self.verbose:
                print('Time limit exceed')
            raise KeyboardInterrupt
