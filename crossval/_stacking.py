import pandas as pd
import numpy as np

from ._crossval import cross_val, cross_val_pred


#__all__ = ['stacking', 'StackingTransformer']



def stacking(estimators, cv, X, y, groups=None, X_new=None, test_avg=True,
             voting='auto', method='predict', n_jobs=None, verbose=0):

    oof_preds = []
    new_preds = []
    est_names = []

    # Fit & predict
    for estimator in estimators:

        oof_pred, new_pred = cross_val_pred(estimator, cv, X, y, groups,
            X_new, test_avg, voting, method, n_jobs, verbose)

        oof_preds.append(oof_pred)
        new_preds.append(new_pred)

        name = _extract_est_name(estimator)
        est_names.append(name)

    # Concat predictions
    oof_stack = pd.concat(oof_preds, axis=1)
    new_stack = pd.concat(new_preds, axis=1)

    # Columns renaming
    oof_stack.columns = est_names
    new_stack.columns = est_names

    return oof_stack, new_stack



def _extract_est_name(estimator):

    name = estimator.__class__.__name__

    if name is 'Pipeline':
        # FIXME: more adequate pipeline name extraction
        inner_estimator = estimator.steps[-1][1]
        return _extract_est_name(inner_estimator)

    else:
        for etype in ['Regressor', 'Classifier', 'Ranker']:
            if name.endswith(etype):
                name = name[:-len(etype)]

        return name
