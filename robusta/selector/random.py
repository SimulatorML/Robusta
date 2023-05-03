from typing import Optional, Callable, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.random import check_random_state

from . import _WrappedSelector, _WrappedGroupSelector


class RandomSelector(_WrappedSelector):
    """
    Random feature selector for sampling and evaluating randomly choosen
    feature subsets of specified size.


    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.

    cv : int, cross-validation generator or iterable
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.

    scoring : string, callable or None, optional, default: None
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only a single value.
        If None, the estimator's default scorer (if available) is used.

    min_features, max_features : int or float
        Minimum & maximum number of features. If float, interpreted as
        percentage of total number of features. <max_features> must be greater
        or equal to the <min_features>.

    max_iter : int or None
        Maximum number of iterations. None for no limits. Use <max_time>
        or Ctrl+C for KeyboardInterrupt to stop optimization in this case.

    max_time : float or None
        Maximum time (in seconds). None for no limits. Use <max_iter>
        or Ctrl+C for KeyboardInterrupt to stop optimization in this case.

    weights : {'binomal', 'uniform'}
        Probability for subset sizes:

            - 'uniform': each # of features selected with equal probability
            - 'binomal': each # of features selected with probability, which
            proportional to # of different subsets of given size (binomal
            coefficient nCk, where n - total # of features, k - subset size)

    random_state : int
        Random state for subsets generator

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int, optional (default=1)
        Verbosity level


    Attributes
    ----------
    features_ : list of string
        Feature names

    n_features_ : int
        Total number of features

    min_features_, max_features_ : int
        Minimum and maximum subsets size

    weights_ : Series
        Subset sizes weights (not normalized)

    rstate_ : object
        Random state instance

    trials_ : DataFrame
        All evaluated subsets:

            - 'subset': subset of feature names
            - 'score': average cross-validation score
            - 'time': fitting time

    best_iter_: int
        Best trial's index

    best_score_: float
        Best trial's score

    best_subset_: list of string
        Best subset of features

    total_time_: float
        Total optimization time (seconds)

    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cv: int = 5,
        scoring: Optional[Union[str, Callable]] = None,
        max_iter: int = 20,
        max_time: Optional[int] = None,
        min_features: float = 0.5,
        max_features: float = 0.9,
        weights: str = "uniform",
        n_jobs: int = -1,
        random_state: int = 0,
        verbose: int = 1,
        n_digits: int = 4,
        cv_kwargs: Optional[dict] = None,
    ):
        if cv_kwargs is None:
            cv_kwargs = {}
        self.estimator = estimator
        self.min_features = min_features
        self.max_features = max_features
        self.max_iter = max_iter
        self.max_time = max_time
        self.weights = weights

        self.cv = cv
        self.cv_kwargs = cv_kwargs
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.verbose = verbose
        self.n_digits = n_digits

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None
    ) -> "RandomSelector":
        """
        Fits the random selector to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        y : pd.Series
            Target variable.

        groups : pd.Series, default=None
            Group labels for grouping in stratified sampling.

        Returns
        -------
        self : RandomSelector
            Fitted random selector.
        """

        # Prepare data and set up selector
        self._fit_start(X)

        # Fit selector to data
        self._fit(X, y, groups)

        return self

    def partial_fit(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None
    ) -> "RandomSelector":
        """
        Fits the RandomSelector model to the given dataset.

        Parameters
        ----------
        X : pd.DataFrame
            The input dataset to fit.

        y : pd.Series
            The target variable.

        groups : pd.Series, default=None
            The groups of samples used for splitting the dataset into train/test set.

        Returns
        -------
        self : RandomSelector
            The fitted RandomSelector model.
        """

        # Prepare data and set up selector
        self._fit_start(X, partial=True)

        # Partially fit selector to data
        self._fit(X, y, groups)

        return self

    def _fit(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None
    ) -> "RandomSelector":
        """
        Fits the RandomSelector model by randomly selecting subsets of features according to specified weights,
        and evaluating their performance on the given data.

        Parameters
        ----------
        X : DataFrame
            the input data with shape (n_samples, n_features)
        y : Series
            the target data with shape (n_samples,)
        groups : pd.Series, optional
            group labels of samples used for group-wise cross-validation.
            Only used if `cv='group'`. Defaults to None.

        Returns
        -------
        self : object
            RandomSelector: the fitted RandomSelector model
        """

        # Loop until interrupted by user
        while True:
            try:
                # Randomly select number of features to include
                k = weighted_choice(self.weights_, self.rstate_)

                # Randomly select a subset of features
                subset = self.features_.sample(size=k, random_state=self.rstate_)

                # Evaluate performance of selected subset
                self.eval_subset(subset, X, y, groups)

            except KeyboardInterrupt:
                # If user interrupts, stop fitting
                break

        return self

    def _fit_start(self, X: pd.DataFrame, partial: bool = False) -> "RandomSelector":
        """
        Fits the RandomSelector model to the given dataset in a partial manner.

        Parameters
        ----------
        X : pd.DataFrame
            The input dataset to fit.

        y : pd.Series
            The target variable.

        groups : pd.Series, default=None
            The groups of samples used for splitting the dataset into train/test set.

        Returns
        -------
        self : RandomSelector
            The fitted RandomSelector model.
        """
        if not partial:
            # If not re-using previously computed weights, reset list of trials
            self._reset_trials()

        if not partial and hasattr(self, "random_state"):
            self.rstate_ = check_random_state(self.random_state)

        # Set the list of available features
        self._set_features(X)

        weights_vals = ["uniform", "binomal"]

        if self.weights == "binomal":
            # Generate binomial weights if specified
            self.weights_ = binomal_weights(
                self.min_features_, self.max_features_, self.n_features_
            )

        elif self.weights == "uniform":
            # Generate uniform weights if specified
            self.weights_ = uniform_weights(self.min_features_, self.max_features_)
        else:
            # Raise error if weights value is not valid
            raise ValueError(f"<weights> must be from {weights_vals}")

        return self

    def get_subset(self):
        """
        Return the subset of selected features.

        Returns
        -------
        Union[List[int], np.ndarray]:
            A list or NumPy array of the indices of the selected features.
        """

        if hasattr(self, "best_subset_"):
            # If using the best subset and the attribute `best_subset_` exists, return the best subset
            return self.best_subset_
        else:
            # If the model is not fitted or no subset is available, raise an error
            model_name = self.__class__.__name__
            raise NotFittedError(f"{model_name} is not fitted")


class GroupRandomSelector(_WrappedGroupSelector, RandomSelector):
    pass


fact = lambda x: x * fact(x - 1) if x else 1


def nCk(n: int, k: int) -> int:
    """
    Calculate the binomial coefficient n choose k.

    Parameters
    ----------
    n : int
        The total number of items.
    k : int
        The number of items to choose.

    Returns
    -------
    num_ways : int
        The number of ways to choose k items from n items.

    """
    return fact(n) // fact(k) // fact(n - k)


def binomal_weights(k_min: int, k_max: int, n: int) -> pd.Series:
    """
    Calculate the binomial weights for all values of k between k_min and k_max.

    Parameters
    ----------
    k_min : int
        The minimum value of k.
    k_max : int
        The maximum value of k.
    n : int
        The total number of items.

    Returns
    -------
    binomial_weight : pd.Series
        A series of binomial weights for all values of k between k_min and k_max.

    """
    # Generate a range of k values
    k_range = range(k_min, k_max + 1)

    # Calculate the binomial coefficient for each value of k
    C = [nCk(n, k) for k in k_range]

    # Return a Pandas Series with the binomial weights as values and k values as index
    return pd.Series(C, index=k_range).sort_index()


def uniform_weights(k_min: int, k_max: int) -> pd.Series:
    """
    Calculate uniform weights for all values of k between k_min and k_max.

    Parameters
    k_min : int
        The minimum value of k.
    k_max : int
        The maximum value of k.

    Returns
    -------
    uniform_weight : pd.Series
        A series of uniform weights for all values of k between k_min and k_max.

    """
    # Generate a range of k values
    k_range = range(k_min, k_max + 1)

    # Return a Pandas Series with the value 1 for each value of k and k values as index
    return pd.Series(1, index=k_range).sort_index()


def weighted_choice(weights: pd.Series, rstate: np.random.RandomState) -> int:
    """
    Choose a random value of k based on the given weights.

    Parameters
    ----------
    weights : Series
        A series of weights for each value of k.
    rstate : RandomState
        A random number generator.

    Returns
    -------
    random_value : int
        A random value of k based on the given weights.

    """
    # Generate a random number between 0 and the sum of the weights
    rnd = rstate.uniform() * weights.sum()

    # Iterate over the weights and subtract each weight from the random number
    # If the random number becomes less than or equal to 0, return the corresponding k value
    for i, w in weights.items():
        rnd -= w
        if rnd <= 0:
            return i
