import numpy as np
from pytest import raises
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from robusta.importance import get_importance


def test_get_importance():
    # Test LinearRegression
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([6, 15, 24])
    lr = LinearRegression()
    lr.fit(X, y)
    imp = get_importance(lr)

    assert np.allclose(imp, np.array([1., 1., 1.]))

    # Test RandomForestRegressor
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([6, 15, 24])
    rfr = RandomForestRegressor(random_state=42)
    rfr.fit(X, y)
    imp = get_importance(rfr)

    assert np.allclose(imp, np.array([0.35588235, 0.35882353, 0.28529412]))

    # Test AttributeError
    with raises(AttributeError):
        get_importance(LinearRegression())  # LinearRegression has no feature_importances_ or coef_
