import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone


__all__ = ['FeatureUnion']




class FeatureUnion(BaseEstimator, TransformerMixin):
    '''Concatenates results of multiple transformer objects.

    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Parameters of the transformers may be set using its name and the parameter
    name separated by a '__'. A transformer may be replaced entirely by
    setting the parameter with its name to another transformer,
    or removed by setting to 'drop' or ``None``.

    Parameters
    ----------
    transformers : list of (string, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer.

    Attributes
    ----------

    named_transformers_ : Bunch object, a dictionary with attribute access
        Access the fitted transformer by name.

    '''
    def __init__(self, transformers):
        self.transformers = transformers


    def fit(self, X, y=None):
        """Fit all transformers using X.

        Parameters
        ----------
        X : DataFrame of shape [n_samples, n_features]
            Input data, of which specified subsets are used to fit the transformers.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self : FeatureUnion
            This estimator

        """
        self.named_transformers_ = {}

        for name, transformer in self.transformers:
            fitted_transformer = clone(transformer).fit(X, y)
            self.named_transformers_[name] = fitted_transformer

        return self


    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : DataFrame of shape [n_samples, n_features]
            Input data, of which specified subsets are used to fit the transformers.

        Returns
        -------
        Xt : DataFrame, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.

        """
        Xt_list = []
        for transformer in self.named_transformers_.values():
            Xt_list.append(transformer.transform(X))

        Xt = pd.concat(Xt_list, axis=1)
        return Xt


    def fit_transform(self, X, y=None):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : DataFrame of shape [n_samples, n_features]
            Input data, of which specified subsets are used to fit the transformers.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        Xt : DataFrame, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.

        """
        return self.fit(X, y).transform(X)
