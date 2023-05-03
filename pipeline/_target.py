from typing import Optional, Callable

import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone


class TransformedTargetRegressor(BaseEstimator, RegressorMixin):
    """
    Meta-estimator to regress on a transformed target.
    Useful for applying a non-linear transformation in regression problems. This
    transformation can be given as a Transformer such as the QuantileTransformer
    or as a function and its inverse such as ``log`` and ``exp``.

    The computation during ``fit`` is::

        regressor.fit(X, func(y))

    The computation during ``predict`` is::

        inverse_func(regressor.predict(X))

    Parameters
    ----------
    regressor : object, default=LinearRegression()
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.

    func : function, optional
        Function to apply to ``y`` before passing to ``fit``. Cannot be set at
        the same time as ``transformer``. The function needs to return a
        2-dimensional array. If ``func`` is ``None``, the function used will be
        the identity function.

    inverse_func : function, optional
        Function to apply to the prediction of the regressor. Cannot be set at
        the same time as ``transformer`` as well. The function needs to return
        a 2-dimensional array. The inverse function is used to return
        predictions to the same space of the original training labels.

    Attributes
    ----------
    regressor_ : object
        Fitted regressor.

    """

    def __init__(
        self,
        regressor: Optional[RegressorMixin] = None,
        func: Optional[Callable] = None,
        inverse_func: Optional[Callable] = None,
    ):
        self.regressor = regressor
        self.func = func
        self.inverse_func = inverse_func

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TransformedTargetRegressor":
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : Series, shape (n_samples,)
            Target values.

        Returns
        -------
        self :
            object

        """

        # Check if the input data has index and set attributes
        if hasattr(X, "index"):
            self.return_df = True
            self.y_name = y.name

        # Apply transformation to target variable
        y = self.func(y)

        # Fit the regressor to the transformed target variable
        self.regressor_ = clone(self.regressor).fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict using the base regressor, applying inverse.

        The regressor is used to predict and the ``inverse_func`` is applied
        before returning the prediction.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_pred : Series, shape (n_samples,)
            Target values.

        """

        # Predict using the fitted regressor
        y = self.regressor_.predict(X)

        # Apply inverse transformation to the predicted values
        y = self.inverse_func(y)

        # If input data has index, return a pandas Series with the same index and name as the target variable
        if self.return_df:
            y = pd.Series(y, name=self.y_name, index=X.index)

        return y

    @property
    def feature_importances_(self):
        """
        Return the feature importances of the fitted regressor.

        Returns
        -------
        feature_importances_ : array, shape (n_features,)
            The feature importances. The higher the value, the more important the feature.
        """
        return self.regressor_.feature_importances_

    @property
    def coef_(self):
        """
        Return the coefficients of the fitted regressor.

        Returns
        -------
        coef_ : array, shape (n_features,)
            The coefficients of the fitted regressor. Each coefficient represents the
            change in the target variable for a one-unit change in the corresponding feature.
        """
        return self.regressor_.coef_
