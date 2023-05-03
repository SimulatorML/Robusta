import numpy as np
from sklearn.base import BaseEstimator


def get_importance(model: BaseEstimator) -> np.ndarray:
    """
    Returns the feature importance's of a model.

    Parameters
    ----------
    model : BaseEstimator
        The model for which to compute feature importance's.

    Returns
    -------
    imp : ndarray
        An array of feature importance's.
    """

    # Get the name of the model's class
    name = model.__class__.__name__

    # If the model has feature_importance's_, return them
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_

    # If the model has coef_, return the absolute values of the coefficients
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)

    # If the model has neither feature_importance's_ nor coef_, raise an AttributeError
    else:
        msg = "<{}> has neither <feature_importance's_>, nor <coef_>".format(name)
        raise AttributeError(msg)

    return imp
