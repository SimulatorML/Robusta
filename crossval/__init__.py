from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict
from ._crossval import cross_val, cross_val_pred


__all__ = [
    # sklearn
    'cross_validate', # score, time, estimator
    'cross_val_score', # score
    'cross_val_predict', # oof_pred (NOTE: don't work for NK-Folds)

    # robusta
    'cross_val', # score, time, estimator, preds
    'cross_val_pred', # oof_pred, new_pred
]
