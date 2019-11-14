import numpy as np
import pandas as pd

__all__ = [
    'blend_objective',
]


def blend_objective(y_true, y_pred, coef=[0.25, 0.5, 0.2, 0.05]):
    '''Weighted average of (fair, huber, L2, L1) losses

    Source: https://www.kaggle.com/scaomath/lgb-giba-features-qm9-custom-objective-in-python
    '''

    # fair
    c = 0.5
    residual = y_pred - y_true
    grad = c * residual /(np.abs(residual) + c)
    hess = c ** 2 / (np.abs(residual) + c) ** 2

    # huber
    delta = 1.2
    scale = 1 + (residual / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad_huber = residual / scale_sqrt
    hess_huber = 1 / scale / scale_sqrt

    # L2 (rmse)
    grad_rmse = residual
    hess_rmse = 1.0

    # L1 (mae)
    grad_mae = np.array(residual)
    grad_mae[grad_mae > 0] = 1.
    grad_mae[grad_mae <= 0] = -1.
    hess_mae = 1.0

    return coef[0] * grad + coef[1] * grad_huber + coef[2] * grad_rmse + coef[3] * grad_mae, \
           coef[0] * hess + coef[1] * hess_huber + coef[2] * hess_rmse + coef[3] * hess_mae
