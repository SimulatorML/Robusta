import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model._base import LinearModel


def non_negative_garotte(X: pd.DataFrame,
                         y: pd.Series,
                         alpha: float,
                         tol: float = 1e-6,
                         max_iter: int = 1000):
    """
    Implementation of the non-negative garotte method for linear regression.

    Parameters
    ----------
    X : pd.DataFrame
        The input data.
    y : pd.Series
        The target values.
    alpha : float
        Regularization strength; must be a positive float. Larger values correspond to stronger regularization.
    tol : float, optional, default=1e-6
        The tolerance for the optimization solver.
    max_iter : int, optional, default=1000
        The maximum number of iterations for the optimization solver.

    Returns
    -------
    tuple:
        tuple of three arrays:
            - The estimated coefficients obtained by applying non-negative garotte.
            - The shrinkage coefficients.
            - The residual sum of squares.
    """

    # Ordinart Least Squares coefficients
    coef_ols = LinearRegression(fit_intercept=False).fit(X, y).coef_
    X = X * coef_ols[np.newaxis, :]

    # Shrunken betas
    shrink_coef = Lasso(alpha=alpha, fit_intercept=False, positive=True, tol=tol, max_iter=max_iter).fit(X, y).coef_
    coef = coef_ols * shrink_coef

    # Residual Sum of Squares
    rss = np.sum((y - np.dot(X, coef)) ** 2)
    return coef, shrink_coef, rss


class NNGRegressor(LinearModel):
    """
    Non-Negative Garrote Regressor

    Code source : https://gist.github.com/agramfort/2351057

    Ref:
    Breiman, L. (1995), "Better Subset Regression Using the Nonnegative
    Garrote," Technometrics, 37, 373-384. [349,351]

    Parameters
    ----------
    alpha : float, optional (default 1e-3)
        Constant that multiplies the L1 term. Defaults to 1.0. alpha = 0 is
        equivalent to an ordinary least square,
        solved by the LinearRegression object. For numerical reasons, using
        alpha = 0 with the Lasso object is not
        advised. Given this, you should use the LinearRegression object.

    fit_intercept : boolean, optional (default True)
        Whether to calculate the intercept for this model. If set to False, no
        intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized
        before regression by subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        sklearn.preprocessing.StandardScaler before calling fit on an estimator
        with normalize=False.

    tol : float, optional (default: 1e-6)
        The tolerance for the optimization: if the updates are smaller than
        tol, the optimization code checks
        the dual gap for optimality and continues until it is smaller than tol.

    max_iter : int, optional (default: 1000)
        The maximum number of iterations.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    Attributes
    ----------
    coef_ : array, shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    intercept_ : array
        Independent term in the linear model.

    """
    def __init__(self,
                 alpha: float = 1e-3,
                 fit_intercept: bool = True,
                 normalize: bool = False,
                 tol: float = 1e-4,
                 max_iter: int = 1000,
                 copy_X: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol

    def fit(self,
            X: np.ndarray,
            y: np.ndarray) -> 'NNGRegressor':
        """
        Fits the Non-negative regression model to the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit the model. It can be a NumPy array or a Pandas DataFrame.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : NNGRegressor
            The fitted NNGRegressor object.

        """
        X, y, X_mean, y_mean, X_std = self._preprocess_data(X, y,
                                                            self.fit_intercept, self.normalize, self.copy_X)

        self.coef_, self.shrink_coef_, self.rss_ = non_negative_garotte(X, y,
                                                                        alpha=self.alpha, tol=self.tol,
                                                                        max_iter=self.max_iter)

        self._set_intercept(X_mean, y_mean, X_std)
        return self
