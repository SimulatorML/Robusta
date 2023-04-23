import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from robusta.linear_model import non_negative_garotte

def test_non_negative_garotte():
    # Generate some synthetic data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # Test with alpha = 0.01
    alpha = 0.01
    coef, shrink_coef, rss = non_negative_garotte(X, y, alpha)

    # Check that all coefficients are non-negative
    assert np.all(coef >= 0)

    # Check that the shrinkage coefficients are within (0, 1] range
    assert np.all((0 < shrink_coef) & (shrink_coef <= 1))

    # Check that the RSS is smaller than OLS
    coef_ols = LinearRegression(fit_intercept=False).fit(X, y).coef_
    rss_ols = np.sum((y - np.dot(X, coef_ols)) ** 2)

    # Test with alpha = 0 (should give OLS)
    coef, shrink_coef, rss = non_negative_garotte(X, y, alpha=0)

    assert np.allclose(coef, coef_ols)
    assert np.allclose(shrink_coef, np.ones_like(coef))
