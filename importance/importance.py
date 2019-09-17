import pandas as pd
import numpy as np


__all__ = ['get_importance']



def get_importance(model, X=None):
    '''Get <feature_importances_> of <coef_> attrs from fitted estimator.

    Parameters
    ----------
    estimator : object
        Fitted estimator

    X : DataFrame, shape (n_objects, n_features), optional
        If passed, convert <imp> to Series with <X.columns> as index

    Returns
    -------
    imp : array or Series of shape (n_features, )
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

    if hasattr(X, 'columns'):
        imp = pd.Series(imp, index=X.columns)

    return imp
