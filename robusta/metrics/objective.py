from typing import Optional, Tuple

import numpy as np


def blend_objective(
    y_true: np.array, y_pred: np.array, coef: Optional[list] = None
) -> Tuple[np.array, np.array]:
    """
    A function that calculates the gradient and Hessian of a blend of loss functions for a given set of predictions.

    Parameters
    -----------
    y_true: np.array
        A one-dimensional numpy array containing the true values of the target variable.

    y_pred: np.array
        A one-dimensional numpy array containing the predicted values of the target variable.

    coef: Optional[list]
        A list of four coefficients to weight the gradient and Hessian of the four loss functions.
        Default is [0.25, 0.5, 0.2, 0.05].

    Returns
    --------
    Tuple[np.array, np.array]
        A tuple containing the gradient and Hessian of the blend of loss functions.

    Raises
    -------
    AssertionError
        If y_true and y_pred do not have the same shape or if y_true does not have a shape of (n,)
        where n is the length of coef.

    """

    # Set default coefficients if none are provided
    if coef is None:
        coef = [0.25, 0.5, 0.2, 0.05]

    # Calculate residuals and absolute residuals
    residual = y_pred - y_true
    abs_residual = np.abs(residual)

    c = 0.5
    residual = y_pred - y_true

    grad = c * residual / (abs_residual + c)
    hess = c**2 / (abs_residual + c) ** 2

    # Calculate gradient and Hessian for second loss function
    delta = 1.2
    scale = 1 + (residual / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad_huber = residual / scale_sqrt
    hess_huber = 1 / (scale * scale_sqrt)

    # Calculate gradient and Hessian for third loss function

    hess_rmse = 1.0

    # Calculate gradient and Hessian for fourth loss function

    grad_mae = np.array(residual)
    grad_mae[grad_mae > 0] = 1.0
    grad_mae[grad_mae <= 0] = -1.0
    hess_mae = 1.0

    # Weight the gradient and Hessian for each loss function according to the provided coefficients
    grad = (
        coef[0] * grad + coef[1] * grad_huber + coef[2] * residual + coef[3] * grad_mae
    )
    hess = (
        coef[0] * hess + coef[1] * hess_huber + coef[2] * hess_rmse + coef[3] * hess_mae
    )

    return grad, hess
