import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, clone

from robusta.importance import get_importance


__all__ = ['TransformedTargetRegressor']




class TransformedTargetRegressor(BaseEstimator, RegressorMixin):
    """Meta-estimator to regress on a transformed target.
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
    def __init__(self, regressor=None, func=None, inverse_func=None):
        self.regressor = regressor
        self.func = func
        self.inverse_func = inverse_func


    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : Series, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object

        """
        self.y_name = y.name

        self.regressor_ = clone(self.regressor).fit(X, y.apply(self.func))

        return self


    def predict(self, X):
        """Predict using the base regressor, applying inverse.

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
        y_pred = self.regressor_.predict(X)
        y_pred = pd.Series(y_pred, name=self.y_name, index=X.index)
        y_pred = y_pred.apply(self.inverse_func)

        return y_pred

    @property
    def feature_importances_(self):
        return self.regressor_.feature_importances_

    @property
    def coef_(self):
        return self.regressor_.coef_
