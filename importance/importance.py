from robusta.model import extract_model, extract_model_name

import pandas as pd




def extract_importance(estimator, X=None):
    '''Extract <feature_importances_> of <coef_> attrs from nested estimator
    (currently used for TransformedTargetRegressor and Pipeline).

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

    """

    '''
    model = extract_model(estimator)

    if hasattr(model, 'feature_importances_'):
        attr = 'feature_importances_'
    elif hasattr(model, 'coef_'):
        attr = 'coef_'
    else:
        name = extract_model_name(model)
        msg = "<{}> has neither <feature_importances_>, nor <coef_>".format(name)
        raise AttributeError(msg)

    imp = getattr(model, attr)

    if hasattr(X, 'columns'):
        imp = pd.Series(imp, index=X.columns)

    return imp
