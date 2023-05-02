from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import TransformerMixin, clone
from sklearn.pipeline import _name_estimators
from sklearn.utils.metaestimators import _BaseComposition


class FeatureUnion(_BaseComposition, TransformerMixin):
    """
    Concatenates results of multiple transformer objects.

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
    """

    def __init__(
        self,
        transformers: list,
        n_jobs: Optional[int] = None,
    ):
        self.named_transformers_ = None
        self.transformers = transformers
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "FeatureUnion":
        """
        Fit all transformers using X.

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

        # List of the transformers
        names = [name for name, _ in self.transformers]

        # Create a list of transformers fitted in parallel
        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit)(clone(transformer), X, y)
            for _, transformer in self.transformers
        )

        # Create a dictionary of fitted transformers using the transformer names as keys
        self.named_transformers_ = dict(zip(names, transformers))

        # Return the FeatureUnion instance
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform X separately by each transformer, concatenate results.

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

        # Create a list of transformed data from each fitted transformer, in parallel
        Xt_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._transform)(transformer, X)
            for transformer in self.named_transformers_.values()
        )

        # Concatenate the transformed data into a single DataFrame
        Xt = pd.concat(Xt_list, axis=1)
        return Xt

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit & transform X separately by each transformer, concatenate results.

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

        # Extract names of transformers
        names = [name for name, _ in self.transformers]

        # Execute the transformers in parallel
        paths = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_transform)(clone(transformer), X, y)
            for _, transformer in self.transformers
        )

        # Collect fitted transformers and transformed data
        transformers, Xt_list = zip(*paths)

        # Create a mapping from transformer names to fitted transformers
        self.named_transformers_ = dict(zip(names, transformers))

        # Concatenate transformed data horizontally
        Xt = pd.concat(Xt_list, axis=1)
        return Xt

    @staticmethod
    def _fit_transform(transformer: object, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Helper function to fit and transform a single transformer.

        Parameters
        ----------
        transformer : object with a 'fit_transform' method
            The transformer to be applied to the input data.
        X : pd.DataFrame
            The input data to be transformed.
        y : pd.Series
            Target values (None if TransformerPipeline is used for a preprocessor).

        Returns
        -------
        Tuple
            A tuple containing the fitted transformer and the transformed data.
        """

        # Fit and transform the data using the transformer
        Xt = transformer.fit_transform(X, y)

        # Return the fitted transformer and transformed data as a tuple
        return transformer, Xt

    def _fit(self, transformer: object, X: pd.DataFrame, y: pd.Series) -> object:
        """
        Helper function to fit a single transformer.

        Parameters
        ----------
        transformer : object with a 'fit' method
            The transformer to be fitted.
        X : pd.DataFrame
            The input data to be transformed.
        y : pd.Series, optional
            Target values (None if TransformerPipeline is used for a preprocessor).

        Returns
        -------
        object
            The fitted transformer.
        """
        return transformer.fit(X, y)

    def _transform(self, transformer: object, X: pd.DataFrame) -> np.array:
        """
        Helper function to transform the data using a single transformer.

        Parameters
        ----------
        transformer : object with a 'transform' method
            The transformer to be applied to the input data.
        X : pd.DataFrame
            The input data to be transformed.

        Returns
        -------
        array-like or sparse matrix, shape (n_samples, n_features)
            The transformed data.
        """
        return transformer.transform(X)

    def get_params(self, deep: bool = True) -> dict:
        """
        Get the parameters of the transformer pipeline.

        Parameters
        ----------
        deep : bool, default=True
            If True, return the parameters of all transformers.

        Returns
        -------
        dict
            A dictionary of parameter names mapped to their values.
        """
        return self._get_params("transformers", deep=deep)

    def set_params(self, **kwargs) -> "FeatureUnion":
        """
        Set the parameters of the transformer pipeline.

        Parameters
        ----------
        **kwargs : dict
            The parameters to be set.

        Returns
        -------
        TransformerPipeline
            The TransformerPipeline object with the updated parameters.
        """
        self._set_params("transformers", **kwargs)
        return self


def make_union(*transformers, **kwargs) -> "FeatureUnion":
    """
    Construct a FeatureUnion from the given transformers.

    This is a shorthand for the FeatureUnion constructor; it does not require,
    and does not permit, naming the transformers. Instead, they will be given
    names automatically based on their types. It also does not allow weighting.

    Parameters
    ----------
    *transformers :
        list of estimators

    **kwargs :
        Additional params

    Returns
    -------
    f :
        FeatureUnion

    """
    n_jobs = kwargs.pop("n_jobs", None)

    if kwargs:
        # We do not currently support `transformer_weights` as we may want to
        # change its type spec in make_union
        raise TypeError(
            'Unknown keyword arguments: "{}"'.format(list(kwargs.keys())[0])
        )

    return FeatureUnion(_name_estimators(transformers), n_jobs=n_jobs)
