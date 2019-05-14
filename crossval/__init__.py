from sklearn.model_selection import cross_validate, cross_val_score
from ._crossval import cross_val, cross_val_stack


__all__ = [
    # sklearn
    'cross_validate', # score, time, estimator
    'cross_val_score', # score
    'cross_val_predict', # oof_pred

    # robusta
    'cross_val', # score, time, estimator, preds
    'cross_val_stack', # oof_pred, new_pred
    #'stacking', # oof_preds, new_preds
    #'StackingTransformer',
]
