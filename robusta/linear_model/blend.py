from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import get_scorer

_AVG_TYPES = {
    'mean': lambda X, w: X.dot(w),
    'hmean': lambda X, w: 1 / (X ** -1).dot(w),
    'gmean': lambda X, w: np.exp(np.log(X).dot(w)),
}


def check_avg_type(avg_type: str) -> str:
    """
    Checks if the given <avg_type> is valid and returns its full name.
    Raises a ValueError if the <avg_type> is not valid.

    Parameters:
    avg_type (str): the type of average to be checked

    Returns:
    str: the full name of the <avg_type>

    Raises:
    ValueError: if the given <avg_type> is not a valid value
    """

    # Get list of valid <avg_type> values from _AVG_TYPES
    avg_types = list(_AVG_TYPES.keys())

    # Check if <avg_type> is in the list of valid <avg_type> values
    if avg_type in avg_types:
        # Return the full name of <avg_type> from _AVG_TYPES
        return _AVG_TYPES[avg_type]

    else:
        # Raise an error if <avg_type> is not valid, listing the allowed values
        raise ValueError(f"Invalid value '{avg_type}' for <avg_type>. Allowed values: {avg_types}")


class _BaseBlend(LinearModel):
    """
    Base class for blending models. It extends the scikit-learn's `LinearModel` class.

    Parameters
    ----------
    avg_type : str, optional (default='mean')
        The type of averaging method to use. Possible values are 'mean' and 'weighted'.
    scoring : str, callable or None, optional (default=None)
        Scikit-learn compatible scoring metric. If `None`, the model is not optimized.
    opt_func : callable or None, optional (default=None)
        The optimizer function to use. If `None`, `scipy.optimize.minimize` is used.
    **opt_kws : dict, optional
        Additional keyword arguments to pass to the optimizer function.

    Attributes
    ----------
    avg : callable
        The averaging function used. It can be either `np.mean` or `weighted_mean`.
    n_features_ : int
        The number of features in the training data.
    classes_ : array-like of shape (n_classes,)
        The class labels.
    scorer : callable or None
        The scoring function to use.
    opt_kws : dict, optional
        Additional keyword arguments passed to the optimizer function.
    result_ : OptimizeResult or None
        The result of the optimization. None if `scoring` is None.
    """

    def __init__(self,
                 avg_type: str = 'mean',
                 scoring: Optional[str] = None,
                 opt_func: Optional[Callable] = None,
                 **opt_kws):
        self._estimator_type = None
        self.coef_ = None
        self.result_ = None
        self.scorer = None
        self.n_features_ = None
        self.classes_ = None
        self.avg = None
        self.avg_type = avg_type
        self.scoring = scoring
        self.opt_func = opt_func
        self.opt_kws = opt_kws

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            weights: np.ndarray = None) -> '_BaseBlend':
        """
        Fit the blending model.

        Parameters
        ----------
        X : pd.DataFrame
            The training data.
        y : pd.Series
            The target values.
        weights : array-like of shape (n_features,), optional (default=None)
            The weights to use for blending.

        Returns
        -------
        self : object
            Returns self.
        """

        # if it's a classifier, get the unique classes
        if self._estimator_type is 'classifier':
            self.classes_ = np.unique(y)

        # check the averaging type
        self.avg = check_avg_type(avg_type=self.avg_type)

        # get the number of features
        self.n_features_ = X.shape[1]

        # set the weights for blending
        self.set_weights(weights)

        # if there's a scoring metric, perform optimization
        if self.scoring:
            # get the scorer function
            self.scorer = get_scorer(scoring=self.scoring)

            # define the objective function
            objective = lambda w: -self.set_weights(w).score(X, y)

            # if no optimizer function is provided, use SLS
            if self.opt_func is None:
                # use scipy.optimize.minimize as the default
                self.opt_func = minimize
                self.opt_kws = dict(x0=self.get_weights(), method='SLSQP',
                                    options={'maxiter': 1000}, bounds=[(0., 1.)] * self.n_features_,
                                    constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}])

            # optimize the objective to get the blending weights
            self.result_ = self.opt_func(objective, **self.opt_kws)

            # set the weights based on the optimization result
            self.set_weights(self.result_['x'])

        return self

    def set_weights(self,
                    weights: Optional[np.ndarray]) -> '_BaseBlend':
        """
        Set the blending weights.

        Parameters
        ----------
        weights : array-like of shape (n_features,), optional (default=None)
            The weights to use for blending.

        Returns
        -------
        self : object
            Returns self.
        """

        # If no weights are provided, set them all to equal values
        if weights is None:
            self.coef_ = np.ones(self.n_features_) / self.n_features_

        else:
            # Normalize weights to sum to 1
            self.coef_ = np.array(weights) / np.sum(weights)

        # Return self for chaining method calls
        return self

    def get_weights(self) -> np.ndarray:
        """
        Get the blending weights.

        Returns
        -------
        coef_ : array-like of shape (n_features,)
            The blending weights.
        """

        # Return the blending weights
        return self.coef_

    def score(self,
              X: pd.DataFrame,
              y: pd.Series) -> float:

        # Call the scoring function with the current instance and input data
        return self.scorer(self, X, y)

    @property
    def _is_fitted(self) -> bool:
        """
        Check if the blending model is fitted.

        Returns
        -------
        is_fitted : bool
            True if the model is fitted, False otherwise.
        """

        # Check if the model has a 'result_' attribute or if no scoring function is defined
        return hasattr(self, 'result_') or not self.scoring

    @property
    def intercept_(self) -> float:
        """
        Intercept of the linear model.

        Returns
        -------
        intercept_ : float
            Always returns 0.0.
        """

        # The intercept for this model is always 0.0
        return .0

    def _blend(self,
               X: pd.DataFrame) -> np.ndarray:
        """
        Blend the predictions using the blending weights.

        Parameters
        ----------
        X : pd.DataFrame
            The predictions of the base models.

        Returns
        -------
        y_blend : array-like of shape (n_samples,)
            The blended predictions.
        """

        # Compute the weighted average of the input predictions using the current blending weights
        return self.avg(X, self.coef_).values


class BlendRegressor(_BaseBlend, RegressorMixin):
    """
    Blending Estimator for regression

    Parameters
    ----------
    avg_type : string or callable (default='mean')
        Select weighted average function:

            - 'mean': Arithmetic mean
            - 'hmean': Harmonic mean
            - 'gmean': Geometric mean

        If passed callable, expected signature: f(X, weights) -> y.

    scoring : string or None (default=None)
        Objective for optimization. If None, all weights are equal. Otherwise,
        calculate the optimal weights for blending. Differentiable scoring are
        prefered.

    opt_func : function (default=None)
        Optimization function. First argument should take objective to minimize.
        Expected signature: f(objective, **opt_kws). If not passed, but scoring
        is defined, used scipy's <minimize> function with method 'SLSQP'.

        Should return result as dict with key 'x' as optimal weights.

    opt_kws : dict, optional
        Parameters for <opt_func> function.


    Attributes
    ----------
    coef_ : Series, shape (n_features, )
        Estimated weights of blending model.

    n_iters_ : int
        Number of evaluations

    result_ : dict
        Evaluation results
    """

    def predict(self,
                X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for a given input.

        Parameters:
        -----------
        X: pd.DataFrame
            The input data to make predictions on. The shape should be (n_samples, n_features).

        Returns:
        --------
        predictions: np.ndarray
            The predicted labels for the input data. The shape should be (n_samples,).
        """
        return self._blend(X)


class BlendClassifier(_BaseBlend, ClassifierMixin):
    """
    Blending Estimator for classification

    Parameters
    ----------
    avg_type : string or callable (default='mean')
        Select weighted average function:

            - 'mean': Arithmetic mean
            - 'hmean': Harmonic mean
            - 'gmean': Geometric mean

        If passed callable, expected signature: f(X, weights) -> y.

    scoring : string or None (default=None)
        Objective for optimization. If None, all weights are equal. Otherwise,
        calculate the optimal weights for blending. Differentiable scoring are
        prefered.

    opt_func : function (default=None)
        Optimization function. First argument should take objective to minimize.
        Expected signature: f(objective, **opt_kws). If not passed, but scoring
        is defined, used scipy's <minimize> function with method 'SLSQP'.

        Should return result as dict with key 'x' as optimal weights.

    opt_kws : dict, optional
        Parameters for <opt_func> function.


    Attributes
    ----------
    coef_ : Series, shape (n_features, )
        Estimated weights of blending model.

    n_iters_ : int
        Number of evaluations

    result_ : dict
        Evaluation results

    """

    def predict_proba(self,
                      X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for a given input.

        Parameters:
        -----------
        X: pd.DataFrame
            The input data to make predictions on. The shape should be (n_samples, n_features).

        Returns:
        --------
        proba: np.ndarray
            A 2D array of shape (n_samples, 2), where each row represents the probability of the
            corresponding sample belonging to each class. The first column is the probability of
            the negative class, and the second column is the probability of the positive class.
        """
        y = self._blend(X)
        return np.stack(arrays=[1 - y, y], axis=-1)

    def predict(self,
                X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for a given input.

        Parameters:
        -----------
        X: np.ndarray
            The input data to make predictions on. The shape should be (n_samples, n_features).

        Returns:
        --------
        predictions: np.ndarray
            The predicted labels for the input data. The shape should be (n_samples,).
            Each label is either 0 (for the negative class) or 1 (for the positive class).
        """
        y = self.predict_proba(X)
        return np.rint(y[:, 1]).astype(int)
