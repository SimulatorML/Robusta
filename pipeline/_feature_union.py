import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from sklearn.utils.metaestimators import _BaseComposition
from sklearn.base import TransformerMixin, clone
from sklearn.pipeline import _name_estimators


__all__ = ['FeatureUnion', 'make_union']




class FeatureUnion(_BaseComposition, TransformerMixin):
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

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    Attributes
    ----------

    named_transformers_ : Bunch object, a dictionary with attribute access
        Access the fitted transformer by name.

    '''
    def __init__(self, transformers, n_jobs=None, **kwargs):
        self.transformers = transformers
        self.n_jobs = n_jobs


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
        names = [name for name, _ in self.transformers]

        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit)(clone(transformer), X, y)
            for _, transformer in self.transformers)

        self.named_transformers_ = dict(zip(names, transformers))

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
        Xt_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._transform)(transformer, X)
            for transformer in self.named_transformers_.values())

        Xt = pd.concat(Xt_list, axis=1)
        return Xt


    def fit_transform(self, X, y=None):
        """Fit & transform X separately by each transformer, concatenate results.

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
        names = [name for name, _ in self.transformers]

        paths = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_transform)(clone(transformer), X, y)
            for _, transformer in self.transformers)

        transformers, Xt_list = zip(*paths)

        self.named_transformers_ = dict(zip(names, transformers))

        Xt = pd.concat(Xt_list, axis=1)
        return Xt


    def _fit_transform(self, transformer, X, y):
        Xt = transformer.fit_transform(X, y)
        return transformer, Xt


    def _fit(self, transformer, X, y):
        return transformer.fit(X, y)


    def _transform(self, transformer, X):
        return transformer.transform(X)


    def get_params(self, deep=True):
        return self._get_params('transformers', deep=deep)


    def set_params(self, **kwargs):
        self._set_params('transformers', **kwargs)
        return self




def make_union(*transformers, **kwargs):
    """Construct a FeatureUnion from the given transformers.

    This is a shorthand for the FeatureUnion constructor; it does not require,
    and does not permit, naming the transformers. Instead, they will be given
    names automatically based on their types. It also does not allow weighting.

    Parameters
    ----------
    *transformers : list of estimators

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    f : FeatureUnion

    """
    n_jobs = kwargs.pop('n_jobs', None)

    if kwargs:
        # We do not currently support `transformer_weights` as we may want to
        # change its type spec in make_union
        raise TypeError('Unknown keyword arguments: "{}"'.format(list(kwargs.keys())[0]))

    return FeatureUnion(_name_estimators(transformers), n_jobs=n_jobs)
