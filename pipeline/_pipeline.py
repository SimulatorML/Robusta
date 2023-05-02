from typing import Union

import numpy as np
from imblearn import pipeline
from sklearn.pipeline import _name_estimators


class Pipeline(pipeline.Pipeline):
    """
    A scikit-learn Pipeline with additional properties for feature importances and coefficients.
    """

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Return the feature importances of the final estimator in the pipeline.

        Returns:
        -------
        np.ndarray:
            An array of feature importances of the final estimator in the pipeline.
        """
        return self._final_estimator.feature_importances_

    @property
    def coef_(self) -> Union[np.ndarray, None]:
        """
        Return the coefficients of the final estimator in the pipeline.

        Returns:
        -------
        np.ndarray or None:
            If the final estimator has `coef_` attribute, return its value, otherwise None.
        """
        if hasattr(self._final_estimator, "coef_"):
            return self._final_estimator.coef_
        else:
            return None


def make_pipeline(*steps, **kwargs) -> Pipeline:
    """
    Construct a Pipeline from the given estimators.

    This is a shorthand for the Pipeline constructor; it does not require, and
    does not permit, naming the estimators. Instead, their names will be set
    to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of estimators.

    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    Returns
    -------
    p : Pipeline

    """

    # Get the value of the `memory` keyword argument, or None if not provided.
    memory = kwargs.pop("memory", None)

    # Check if any unknown keyword arguments were provided.
    if kwargs:
        raise TypeError(
            'Unknown keyword arguments: "{}"'.format(list(kwargs.keys())[0])
        )

    # Create a new Pipeline object from the given estimators and return it.
    return Pipeline(_name_estimators(steps), memory=memory)
