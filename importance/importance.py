import pandas as pd
import numpy as np


__all__ = ['get_importance']



def get_importance(model):
    '''Get <feature_importances_> of <coef_> attrs from fitted estimator.

    Parameters
    ----------
    model : estimator object
        Fitted estimator

    Returns
    -------
    imp : array of shape (n_features, )
        Feature importances of fitted estimator

    '''
    name = model.__class__.__name__

    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_

    elif hasattr(model, 'coef_'):
        imp = model.coef_

    else:
        msg = "<{}> has neither <feature_importances_>, nor <coef_>".format(name)
        raise AttributeError(msg)

    return imp
