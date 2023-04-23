import numpy as np
import pytest

from robusta.metrics import blend_objective


def test_blend_objective():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([2, 3, 4, 5])
    coef = [0.25, 0.5, 0.2, 0.05]
    grad, hess = blend_objective(y_true, y_pred, coef)
    assert grad.shape == y_true.shape
    assert hess.shape == y_true.shape
    assert grad.dtype == np.float64
    assert hess.dtype == np.float64
    assert isinstance(grad, np.ndarray)
    assert isinstance(hess, np.ndarray)