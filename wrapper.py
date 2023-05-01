import numpy as np
import pandas as pd

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer


class WrappedRegressor(BaseEstimator, ClassifierMixin):
    """
    Wrapper class to transform a regressor into a binary classifier

    Parameters
    ----------
    regressor : estimator object
        The regressor to be wrapped.
    method : {'minmax', None}, default='minmax'
        The method used to transform the regressor's output into class probabilities.
        'minmax' scales the output to [0, 1], while None uses the raw output.

    Attributes
    ----------
    regressor_ : estimator object
        The fitted regressor.
    classes_ : array, shape (n_classes,)
        The classes seen during fit.

    Methods
    -------
    fit(X, y)
        Fit the regressor to the data and binarize the target variable.
    decision_function(X)
        Compute the decision function of the regressor on the input data.
    predict_proba(X)
        Compute class probabilities of the input data.
    predict(X)
        Predict the class of the input data.
    """
    def __init__(self,
                 regressor: BaseEstimator,
                 method: str = 'minmax'):
        self.regressor_ = None
        self.classes_ = None
        self.regressor = regressor
        self.method = method

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series) -> 'WrappedRegressor':
        """
        Fit the regressor to the data and binarize the target variable.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input data.
        y : pandas Series of shape (n_samples,)
            The target variable.

        Returns
        -------
        self : object
            Returns self.
        """
        # Binarize the target variable
        le = LabelBinarizer().fit(y)
        self.classes_ = le.classes_

        # Fit the regressor to the data
        self.regressor_ = clone(self.regressor).fit(X, y)

        return self

    def decision_function(self,
                          X: pd.DataFrame) -> np.array:
        """
        Compute the decision function of the regressor on the input data.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : numpy array of shape (n_samples,)
            The output of the regressor.
        """

        # Compute the output of the regressor
        y = self.regressor_.predict(X)

        # Shift the output so that the decision threshold is 0
        y = y - 0.5
        return y

    def predict_proba(self,
                      X: pd.DataFrame) -> np.ndarray:
        """
        Compute class probabilities of the input data.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : numpy array of shape (n_samples, 2)
            The class probabilities of the input data.
        """

        # Compute the decision function
        y = self.decision_function(X)

        # Scale the output to [0, 1] using the specified method
        if self.method is 'minmax':
            y = y - y.min()
            y = y / y.max()

        # Return the class probabilities
        return np.stack([1 - y, y], axis=-1)

    def predict(self,
                X: pd.DataFrame) -> np.array:
        """
        Predict the class of the input data.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : numpy array of shape (n_samples,)
            The predicted classes of the input data.
        """
        y = self.decision_function(X)
        y = 1 * (y > 0)
        return y

    @property
    def coef_(self) -> np.ndarray:
        """
        Coefficients of the linear model.

        Returns
        -------
        coef_ : numpy array of shape (n_features,) or (n_classes, n_features)
            The coefficients of the linear model.
        """
        return self.regressor_.coef_

    @property
    def feature_importances_(self) -> np.array:
        """
        Feature importances of the model.

        Returns
        -------
        feature_importances_ : numpy array of shape (n_features,)
            The feature importances of the model.
        """
        return self.regressor_.feature_importances_
