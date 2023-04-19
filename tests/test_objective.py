import numpy as np
import pytest

from metrics.objective import blend_objective


@pytest.fixture
def y_true():
    """
    Fixture function to generate the true target values.
    """
    return np.array([1, 2, 3, 4, 5])


@pytest.fixture
def y_pred():
    """
    Fixture function to generate the predicted target values.
    """
    return np.array([2, 2, 3, 3, 5])


def test_blend_objective(y_true, y_pred):
    """
    Test function to test the blend_objective function.

    Parameters:
    -----------
    y_true : ndarray
       Array of true target values.

    y_pred : ndarray
       Array of predicted target values.

    Returns:
    --------
    None

    Raises:
    -------
    ValueError:
       If y_true and y_pred have different shapes.
    """

    coefficient = [0.25, 0.5, 0.2, 0.05]
    grad, hess = blend_objective(y_true=y_true, y_pred=y_pred, coef=coefficient)

    # Test that the shapes of the gradients and Hessians are correct
    assert grad.shape == (5,)
    assert hess.shape == (5,)

    # Test that the values of the gradients and Hessians are correct
    assert np.allclose(grad, np.array([0.71744397, -0.05, -0.05, -0.71744397, -0.05]))
    assert np.allclose(hess, np.array([0.50446602, 1., 1., 0.50446602, 1.]))

    # Test that the function raises a ValueError when y_true and y_pred have different shapes
    with pytest.raises(ValueError):
        blend_objective(y_true[:-1], y_pred)


