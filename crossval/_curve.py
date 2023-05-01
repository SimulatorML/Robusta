from typing import Generator, Callable, Tuple, List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor, XGBClassifier


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Calculate the sigmoid function for a given input.

    Parameters:
    -----------
    x (np.ndarray): The input array.

    Returns:
    --------
    np.ndarray: The output array with values between 0 and 1.

    Examples:
        >>> x = np.array([0, 1, 2])
        >>> sigmoid(x)
        array([0.5, 0.73105858, 0.88079708])
    """
    return 1 / (1 + np.exp(-x))


def _xgb_staged_predict(estimator: XGBRegressor | XGBClassifier,
                        X: pd.DataFrame,
                        max_iter: int = 0,
                        step: int = 1) -> None:
    """
    Generate predictions of the input data for each stage of the XGBoost model.

    Parameters:
    -----------
    estimator : XGBRegressor or XGBClassifier
        The XGBoost model to use for prediction.
    X : pd.DataFrame
        The input data to predict.
    max_iter : int, default 0
        The maximum number of iterations for the model.
    step : int, default 1
        The number of iterations between consecutive predictions.

    Returns:
    --------
    None
    """

    # Get the booster object of the estimator
    booster = estimator.get_booster()

    # Generate leaf indices for each instance of the input data up to a maximum iteration
    leafs = estimator.apply(X, max_iter)

    # Get the shape of the leaf indices array
    M = leafs.shape[0]
    N = leafs.shape[1]

    # Generate an array of tree indices for each stage of the model
    trees = np.mgrid[0:M, 0:N][1]

    return None


def _lgb_staged_predict(estimator: LGBMRegressor | LGBMClassifier,
                        X: pd.DataFrame,
                        max_iter: int = 0,
                        step: int = 1) -> np.ndarray:
    """
    Generate predictions of the input data for each stage of the LightGBM model.

    Parameters:
    -----------
    estimator : LGBMRegressor or LGBMClassifier
        The LightGBM model to use for prediction.
    X : pd.DataFrame
        The input data to predict.
    max_iter : int, default 0
        The maximum number of iterations for the model.
    step : int, default 1
        The number of iterations between consecutive predictions.

    Yields:
        np.ndarray: The predicted values for each stage.

    Examples:
        >>> from lightgbm import LGBMClassifier
        >>> from sklearn.datasets import make_classification
        >>> X_train, y_train = make_classification(n_samples=1000, n_features=10, random_state=42)
        >>> clf = LGBMClassifier(n_estimators=100, random_state=42)
        >>> clf.fit(X_train, y_train)
        >>> for prediction in _lgb_staged_predict(clf, X_train):
        ...     print(prediction)
    """

    # Get the booster and maximum number of stages to generate predictions for
    booster = estimator.booster_
    max_iter = max_iter if max_iter else booster.num_trees()

    # Predict the leaf indices for each sample in X
    leafs = booster.predict(X,
                            pred_leaf=True,
                            num_iteration=max_iter)

    # Determine the number of samples and leaves
    M, N = leafs.shape

    # Create an array of tree indices for each sample
    trees = np.mgrid[0:M, 0:N][1]

    # Create a dictionary mapping leaf indices to the corresponding output values for each stage
    mapper = {}
    for i in range(estimator.n_estimators):
        for j in range(estimator.num_leaves):
            mapper[i, j] = booster.get_leaf_output(i, j)

    # Map leaf indices to output values for each stage
    preds = np.vectorize(lambda i, j: mapper[i, j])(trees, leafs).T

    # Cumulatively sum predicted values over the stages and yield predictions for each step
    preds = preds.cumsum(axis=0)[np.arange(step, max_iter + step, step) - 1]

    # Apply sigmoid function to predicted values if estimator is a classifier
    if estimator._estimator_type == 'classifier':
        preds = sigmoid(preds)

        for pred in preds:
            # Yield probabilities for each class
            yield np.vstack([1 - pred, pred]).T

    # Apply sigmoid function to predicted values if estimator is a regressor
    elif estimator._estimator_type == 'regressor':
        # Yield predicted values for each stage
        for pred in preds:
            yield pred


def _cat_staged_predict(estimator: CatBoostRegressor | CatBoostClassifier,
                        X: pd.DataFrame,
                        max_iter: int = 0,
                        step: int = 1) -> Generator:
    """
    Generates staged predictions for a fitted CatBoost estimator.

    Parameters
    ----------
    estimator : CatBoostRegressor or CatBoostClassifier
        The fitted CatBoost estimator to generate predictions for.
    X : ndarray of shape (n_samples, n_features)
        The input data to generate predictions for.
    max_iter : int, default 0
        The maximum number of stages to generate predictions for. If set to 0, it will generate predictions for all stages.
    step : int, default 1
        The step size between stages to generate predictions for.

    Yields
    ------
    Generator
        A generator object that yields predicted values or probabilities for each stage of the estimator.

    Examples
    --------
    >>> from catboost import CatBoostRegressor, CatBoostClassifier
    >>> import numpy as np
    >>> from typing import List

    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, 100)

    >>> clf = CatBoostClassifier(iterations=10, depth=2, learning_rate=1, loss_function='Logloss')
    >>> clf.fit(X, y)

    >>> for i, pred in enumerate(_cat_staged_predict(clf, X, max_iter=10, step=1)):
    ...     print(f'Stage {i}: {pred}')
    """

    # Predict classification target at each stage for data.
    if estimator._estimator_type == 'classifier':
        return estimator.staged_predict_proba(X, ntree_end=max_iter, eval_period=step)

    # Predict target at each stage for data.
    elif estimator._estimator_type == 'regressor':
        return estimator.staged_predict(X, ntree_end=max_iter, eval_period=step)


def _get_scores(estimator: BaseEstimator,
                generator: Callable,
                predictor: Callable,
                trn: np.array,
                val: np.array,
                X: pd.DataFrame,
                y: pd.Series,
                scorer: Callable,
                max_iter: int,
                step: int,
                train_score: bool) -> Tuple[List[float], List[float]]:
    """
    Compute training and validation scores for a given estimator, generator, predictor, data, and scorer.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator object to use.
    generator : Callable
        The generator function to use for producing staged predictions.
    predictor : Callable
        The function to use for computing the final predictions.
    trn : np.array
        The indices of the training data.
    val : np.array
        The indices of the validation data.
    X : pd.DataFrame
        The input data to use for making predictions.
    y : pd.Series
        The target variable to use for computing the scores.
    scorer : Callable
        The scoring function to use for computing the scores.
    max_iter : int
        The maximum number of iterations to use for generating staged predictions.
    step : int
        The step size to use for generating staged predictions.
    train_score : bool
        Whether to compute training scores.

    Returns
    ----------
    trn_scores, val_scores : Tuple[List[float], List[float]]
        A tuple containing the training and validation scores.
    """

    # Generate a list of stages to calculate scores for
    stages = np.arange(step, max_iter + step, step)

    # Initialize lists to store the scores
    trn_scores = []
    val_scores = []

    # Calculate scores for the training set if requested
    if train_score:
        X_trn, y_trn = X.iloc[trn], y.iloc[trn]

        # Generate a sequence of predictions for the training set
        S_trn = generator(estimator, X_trn, max_iter, step)

        for _ in stages:
            # Calculate the score for the predicted values and the true values
            trn_scores.append(scorer(predictor, next(S_trn), y_trn))

    if True:
        # Calculate scores for the validation set
        X_val, y_val = X.iloc[val], y.iloc[val]
        S_val = generator(estimator, X_val, max_iter, step)

        for _ in stages:
            # Calculate the score for the predicted values and the true values
            val_scores.append(scorer(predictor, next(S_val), y_val))

    # Return the scores for both sets
    return trn_scores, val_scores


class _StagedClassifier(BaseEstimator):
    """
    A simple classifier that returns the input array as the predicted probabilities and
    classifies instances as 1 if the probability of the positive class is greater than 0.5.

    Attributes:
        _estimator_type (str): The estimator type, which is 'classifier' for this class.

    Methods:
        predict_proba(X: np.array) -> np.array:
            Return the input array as the predicted probabilities.

        predict(X: np.ndarray) -> np.ndarray:
            Classify instances as 1 if the probability of the positive class is greater than 0.5.
    """

    _estimator_type = 'classifier'

    @staticmethod
    def predict_proba(X: np.ndarray) -> np.ndarray:
        """
        Return the input array as the predicted probabilities.

        Parameters
        ----------
        X : np.ndarray
            The input array.

        Returns
        -------
        X : np.ndarray
            The same input array.
        """

        # Return the same input array
        return X

    @staticmethod
    def predict(X: np.ndarray) -> np.ndarray:
        """
        Classify instances as 1 if the probability of the positive class is greater than 0.5.

        Parameters
        ----------
        X : np.ndarray
            The input array.

        Returns
        -------
        y : np.ndarray
            The predicted labels, which are 1 if the probability of the positive class is greater
            than 0.5 and 0 otherwise.

        Examples
        --------
            >>> X = np.array([[0.1, 0.9], [0.5, 0.5], [0.7, 0.3]])
            >>> _StagedClassifier.predict(X)
            array([1, 0, 1])
        """

        # Determine which class has the highest probability
        y = X[:, 1] > 0.5
        y = y.astype(int)

        # Return the predicted class labels
        return y


class _StagedRegressor(BaseEstimator):
    """
    A dummy regressor that returns the input array.

    This is used for the staged prediction of a scikit-learn regressor.

    Attributes:
        _estimator_type (str): The type of the estimator, which is 'regressor' for this class.

    Methods:
        predict: Return the input array.
    """

    _estimator_type = 'regressor'

    @staticmethod
    def predict(X: np.ndarray) -> np.ndarray:
        """
        Return the input array.

        This method is used for the staged prediction of a scikit-learn regressor.

        Parameters
        ----------
        X : np.ndarray
            The input array.

        Returns
        -------
        X : np.ndarray
            The input array.
        """

        # Return the input array
        return X
