import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin




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
        Xt = X[self.use_cols_]
        return Xt


    def inverse_transform(self, X):
        """
        Reverse the transformation operation

        Parameters
        ----------
        X : DataFrame of shape [n_samples, n_selected_features]
            The input samples.

        Returns
        -------
        Xt : DataFrame of shape [n_samples, n_original_features]
            `X` with columns of NaNs inserted where features would have
            been removed by `transform`.

        """
        Xt = pd.DataFrame(index=X.index, columns=self.base_cols_)
        Xt = Xt.astype(self.base_dtypes_)
        Xt[X.columns] = X
        return Xt

    #def __init__(self, estimator, cv=5, scoring=None,
    #             max_time=None, max_trials=None, n_jobs=None,
    #             verbose=1, plot=False):



    #def fit(self, X, y, groups=None):

    #    self._fit_start(X, y, groups)

    #    self._fit(X, y, groups) # defined individualy
    #    self._fit_end()

    #    return self



    #def partial_fit(self, X, y, groups=None):

    #    self._fit_start(X, y, groups, partial_fit=True)

    #    self._fit(X, y, groups) # defined individualy
    #    self._fit_end()

    #    return self
