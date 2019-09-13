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

        scores = crossval_score(estimator, self.cv, X[subset], y, groups,
                                scoring=self.scoring, n_jobs=self.n_jobs)
        score = np.mean(scores)

        if hasattr(self, 'best_score_') is True and (score > self.best_score_) \
        or hasattr(self, 'best_score_') is False:

            self.best_subset_ = subset
            self.best_score_ = score

        return score
