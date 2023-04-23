from typing import Optional, List, Callable, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.random import check_random_state

from . import _WrappedSelector, _WrappedGroupSelector, _check_k_features


def perturb_subset(subset: pd.Series,
                   step: int,
                   random_state: Optional[int] = None,
                   drop_attrs: Optional[List[str]] = None) -> pd.Series:
    """
    Randomly perturb the feature subset by removing a randomly selected set of <step> features
    and adding a different set of <step> features.

    Parameters:
    subset (pd.Series): The subset of features to be perturbed.
    step (int): The number of features to be randomly removed and added.
    random_state (Optional[int]): Seed to ensure reproducibility of random number generation.
    drop_attrs (Optional[List[str]]): Attributes to drop from the resulting perturbed subset.

    Returns:
        pd.Series: A perturbed subset of features.

    Raises:
        None
    """

    # Set default value for drop_attrs if not provided
    if drop_attrs is None:
        drop_attrs = ['score']

    # Check random state and get random numbers
    rstate = check_random_state(random_state)
    update = rstate.choice(subset.features, step, False)

    # Find features to remove and add
    del_list = set(subset) & set(update)
    add_list = set(update) - set(subset)

    # Copy the original subset and update it by removing and adding the features
    subset_ = subset.copy()
    subset_ = subset_.remove(*del_list)
    subset_ = subset_.append(*add_list)

    # Remove the specified attributes from the subset
    for attr in drop_attrs:
        if hasattr(subset_, attr):
            delattr(subset_, attr)

    # Set the parent of the perturbed subset as the original subset
    subset_.parents = (subset,)

    return subset_


class SAS(_WrappedSelector):
    """
    Sequential Feature Selection with Simulated Annealing algorithm.

    Parameters:
        estimator (BaseEstimator): The estimator to use in the selection process.
        cv (int): Number of cross-validation folds. Defaults to 5.
        scoring (Optional[Union[str, Callable]]): Scoring metric or callable to evaluate predictions on the test set.
            If None, the estimator's default scorer is used. Defaults to None.
        min_step (float): The minimum fraction of features to remove at each iteration. Defaults to 0.01.
        max_step (float): The maximum fraction of features to remove at each iteration. Defaults to 0.05.
        min_features (float): The minimum fraction of features to select. Defaults to 0.1.
        max_features (float): The maximum fraction of features to select. Defaults to 0.9.
        max_iter (int): The maximum number of iterations to perform. Defaults to 50.
        temp (float): The starting temperature for simulated annealing. Defaults to 1.0.
        random_state (Optional[int]): Seed for the random number generator. Defaults to None.
        n_jobs (Optional[int]): The number of jobs to run in parallel. Defaults to None.
        verbose (int): Controls the verbosity of the selection process. Defaults to 1.
        n_digits (int): The number of digits to display for floating-point values. Defaults to 4.
        cv_kwargs (Optional[dict]): Additional arguments to pass to the cross-validation function.

    Attributes:
        estimator: The estimator used in the selection process.
        cv: Number of cross-validation folds.
        cv_kwargs: Additional arguments to pass to the cross-validation function.
        scoring: Scoring metric or callable used to evaluate predictions on the test set.
        min_features: The minimum fraction of features to select.
        max_features: The maximum fraction of features to select.
        min_step: The minimum fraction of features to remove at each iteration.
        max_step: The maximum fraction of features to remove at each iteration.
        max_iter: The maximum number of iterations to perform.
        temp: The starting temperature for simulated annealing.
        random_state: Seed for the random number generator.
        verbose: Controls the verbosity of the selection process.
        n_digits: The number of digits to display for floating-point values.
        n_jobs: The number of jobs to run in parallel.
    """
    def __init__(self,
                 estimator: BaseEstimator,
                 cv: int = 5,
                 scoring: Optional[Union[str, Callable]] = None,
                 min_step: float = 0.01,
                 max_step: float = 0.05,
                 min_features: float = 0.1,
                 max_features: float = 0.9,
                 max_iter: int = 50,
                 temp: float = 1.0,
                 random_state: Optional[int] = None,
                 n_jobs: Optional[int] = None,
                 verbose: int = 1,
                 n_digits: int = 4,
                 cv_kwargs: Optional[dict] = None):

        if cv_kwargs is None:
            cv_kwargs = {}
        self.estimator = estimator
        self.cv = cv
        self.cv_kwargs = cv_kwargs
        self.scoring = scoring

        self.min_features = min_features
        self.max_features = max_features
        self.min_step = min_step
        self.max_step = max_step
        self.max_iter = max_iter
        self.temp = temp

        self.random_state = random_state
        self.verbose = verbose
        self.n_digits = n_digits
        self.n_jobs = n_jobs

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            groups: Optional[pd.Series] = None) -> 'SAS':
        """
        Run the SAS algorithm on the input data and select the best feature subset.

        Parameters:
            X (pd.DataFrame): The input data to fit.
            y (pd.Series): The target variable to predict.
            groups (pd.Series): Group labels for the samples, used for grouped cross-validation.

        Returns:
            SAS: The fitted SAS estimator.

        """

        # Prepare data and set up selector
        self._fit_start(X, y, groups)

        # Fit selector to data
        self._fit(X, y, groups)
        return self

    def partial_fit(self,
                    X: pd.DataFrame,
                    y: pd.Series,
                    groups: Optional[pd.Series] = None) -> 'SAS':
        """
        Fits the sas to a partial amount of data.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix.

        y : pd.Series
            Target variable.

        groups : pd.Series, default=None
            Group labels for grouping in stratified sampling.

        Returns:
        --------
        self : SAS
            Partially fitted genetic selector.
        """

        # Partially fit selector to data
        self._fit(X, y, groups)

        return self

    def _fit_start(self,
                   X: pd.DataFrame,
                   y: pd.Series,
                   groups: pd.Series) -> 'SAS':
        """
        Initializes the sas before fitting.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix.

        partial : bool, default=False
            Whether the fitting is partial or full.

        Returns:
        --------
        self : SAS
            Initialized sas.
        """
        # Basic
        self.rstate_ = check_random_state(self.random_state)

        # Set features
        self._set_features(X)
        self._reset_trials()

        # First trial
        k_min = self.min_features_
        k_max = self.max_features_
        k = self.rstate_.choice(range(k_min, k_max + 1))
        subset = self.features_.sample(size=k, random_state=self.rstate_)

        self.eval_subset(subset, X, y, groups)
        self.subset_ = subset

        return self

    def _fit(self,
             X: pd.DataFrame,
             y: pd.Series,
             groups: Optional[pd.Series] = None) -> 'SAS':
        """
        Fits the SAS model to the provided data.

        Parameters:
        -----------
        X : pd.DataFrame
            The input feature dataset.
        y : pd.Series
            The target variable.
        groups : pd.Series
            The groups variable, used for cross-validation.

        Returns:
        --------
        SAS
            The fitted SAS object.
        """

        while self.n_iters_ < self.max_iter:
            try:
                # Pertrub the current subset
                k_min = self.min_step_
                k_max = self.max_step_
                k = self.rstate_.choice(range(k_min, k_max + 1))
                subset = perturb_subset(self.subset_, k, self.rstate_)

                # Evaluate perfomance
                self.eval_subset(subset, X, y, groups)
                old_score = self.subset_.score
                new_score = subset.score

                if new_score > old_score:
                    self.subset_ = subset

                else:
                    # Acceptance probability
                    temp = self.temp * self.max_iter / self.n_iters_
                    diff = (old_score - new_score) / abs(old_score)
                    prob = np.exp(-diff / temp)

                    if self.rstate_.rand() < prob:
                        self.subset_ = subset

            except KeyboardInterrupt:
                break

        return self

    @property
    def min_step_(self) -> float:
        """
        Property that checks if `min_step` is within the range [0, 1].
        If it is a float, it returns it directly, otherwise it computes the
        number of features to be removed based on the input percentage.

        Returns:
            min_step (float): The validated minimum fraction of features to
            remove at each iteration.
        """

        # Ensure min_step is within the range [0, 1]
        min_step = _check_k_features(self.min_step,
                                     self.n_features_,
                                     'min_step')
        return min_step

    @property
    def max_step_(self):
        """
        Property that checks if `max_step` is within the range [0, 1].
        If it is a float, it returns it directly, otherwise it computes the
        number of features to be removed based on the input percentage.

        Returns:
            max_step (float): The validated maximum fraction of features to
            remove at each iteration.
        """

        # Ensure max_step is within the range [0, 1]
        max_step = _check_k_features(self.max_step,
                                      self.n_features_,
                                     'max_step')
        return max_step


class GroupSAS(_WrappedGroupSelector, SAS):
    pass
