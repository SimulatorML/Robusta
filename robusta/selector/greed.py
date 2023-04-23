from typing import Optional, Callable, Union, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.random import check_random_state

from . import _WrappedSelector, _WrappedGroupSelector, _check_k_features
from ..utils import logmsg


class GreedSelector(_WrappedSelector):
    """
    Greed Forward/Backward Feature Selector

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

    forward : boolean (default=True)
        Whether to start from empty set or full set of features:

            - If <forward> is True, add feature on each step
            - If <forward> is False, drop feature on each step

    floating : boolean (default=False)
        Whether to produce step back on each round (if increases score!):

            - If <forward> is True, drop feature on each step
            - If <forward> is False, drop feature on each step

    k_features : int or float (default=0.5)
        Number of features to select. If float, interpreted as percentage
        of total # of features:

            - If <forward> is True, <k_features> is maximum # of features.
            - If <forward> is False, <k_features> is minimum # of features.

    max_iter : int or None
        Maximum number of iterations. None for no limits. Use <max_time>
        or Ctrl+C for KeyboardInterrupt to stop optimization in this case.

    max_time : float or None
        Maximum time (in seconds). None for no limits. Use <max_iter>
        or Ctrl+C for KeyboardInterrupt to stop optimization in this case.

    use_best : bool (default=True)
        Whether to use subset with best score or last selected subset.

    random_state : int or None (default=0)
        Random seed for permutations in PermutationImportance.
        Ignored if <importance_type> set to 'inbuilt'.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int, optional (default=1)
        Verbosity level

    n_digits : int (default=4)
        Verbose score(s) precision

    """

    def __init__(self,
                 estimator: BaseEstimator,
                 cv: int = 5,
                 scoring: Optional[Union[str, Callable]] = None,
                 forward: bool = True,
                 floating: bool = False,
                 k_features: float = 0.5,
                 max_time: Optional[int] = None,
                 use_best: bool = True,
                 random_state: int = 0,
                 n_jobs: Optional[int] = None,
                 verbose: int = 1,
                 n_digits: int = 4,
                 cv_kwargs: Optional[dict] = None):

        if cv_kwargs is None:
            cv_kwargs = {}
        self.estimator = estimator
        self.k_features = k_features
        self.forward = forward
        self.floating = floating
        # self.max_candidates = max_candidates # TODO
        self.max_time = max_time
        self.use_best = use_best

        self.cv = cv
        self.cv_kwargs = cv_kwargs
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.verbose = verbose
        self.n_digits = n_digits

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            groups: Optional[pd.Series] = None) -> 'GreedSelector':
        """
        Fits the genetic selector to the data.

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
        self : GeneticSelector
            Fitted genetic selector.
        """

        # Prepare data and set up selector
        self._fit_start(X)

        # Fit selector to data
        self._fit(X, y, groups)

        return self

    def partial_fit(self,
                    X: pd.DataFrame,
                    y: pd.Series,
                    groups: Optional[pd.Series] = None) -> 'GreedSelector':
        """
        Fits the GreedSelector model to the given dataset.

        Parameters:
        -----------
        X : pd.DataFrame
            The input dataset to fit.

        y : pd.Series
            The target variable.

        groups : pd.Series, default=None
            The groups of samples used for splitting the dataset into train/test set.

        Returns:
        --------
        self : GreedSelector
            The fitted GreedSelector model.
        """

        # Initialize the GreedSelector model
        self._fit_start(X, partial=True)

        # Fit the GreedSelector model to the given dataset
        self._fit(X, y, groups)

        return self

    def _fit_start(self,
                   X: pd.DataFrame,
                   partial: bool = False) -> 'GreedSelector':
        """
        Fits the GreedSelector model to the given dataset in a partial manner.

        Parameters:
        -----------
        X : pd.DataFrame
            The input dataset to fit.

        y : pd.Series
            The target variable.

        groups : pd.Series, default=None
            The groups of samples used for splitting the dataset into train/test set.

        Returns:
        --------
        self : GreedSelector
            The fitted GeneticSelector model.
        """

        # Set the features to be the columns of the input dataset X
        self._set_features(X)

        # Ensure that the number of features to select is within a valid range
        self.k_features_ = _check_k_features(self.k_features, self.n_features_, 'k_features')

        if not partial:
            # Initialize the random state
            self.rstate_ = check_random_state(self.random_state)

            # Set the subset of features to be the full set of features
            self.subset_ = self.features_.copy()

            # If the forward search method is used, set the subset of features to be an empty list
            if self.forward:
                self.subset_.set_subset([])

            # Reset the trials for the greed algorithm
            self._reset_trials()

        return self

    def _fit(self,
             X: pd.DataFrame,
             y: pd.Series,
             groups: pd.Series) -> 'GreedSelector':
        """
        Fits the Greed Feature Selector model to the provided data.

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
        GreedSelector
            The fitted Greed Feature Selector object.
        """

        if self.forward:
            is_final = lambda subset: len(subset) >= self.k_features_
        else:
            is_final = lambda subset: len(subset) <= self.k_features_

            self.eval_subset(self.subset_, X, y, groups)
            self.score_ = self.subset_.score

        while not is_final(self.subset_):

            # STEP 1. Step Forward/Backward
            if self.verbose:
                logmsg('STEP {}'.format('FORWARD' if self.forward else 'BACKWARD'))

            if self.forward:
                updates = self.features_.remove(*self.subset_)
            else:
                updates = self.subset_

            # Find Next Best Update
            score = -np.inf
            subset = None

            for feature in updates:

                # Include/Exclude Feature
                if self.forward:
                    candidate = self.subset_.append(feature)
                else:
                    candidate = self.subset_.remove(feature)

                candidate.parents = (self.subset_,)

                # Evaluate Candidate
                try:
                    self.eval_subset(candidate, X, y, groups)

                    if candidate.score > score:
                        score = candidate.score
                        subset = candidate

                except KeyboardInterrupt:
                    raise

                except:
                    pass

            # Update Subset
            self.subset_ = subset
            self.score_ = score

            # Stop Criteria
            if not self.floating or is_final(self.subset_):
                continue

            # STEP 2. Step Backward/Forward
            if self.verbose:
                logmsg('STEP {}'.format('BACKWARD' if self.forward else 'FORWARD'))

            if not self.forward:
                updates = self.features_.remove(*self.subset_)
            else:
                updates = self.subset_

            # Find Next Best Update
            score = -np.inf
            subset = None

            for feature in updates:

                # Exclude/Include Feature
                if not self.forward:
                    candidate = self.subset_.append(feature)
                else:
                    candidate = self.subset_.remove(feature)

                candidate.parents = (self.subset_,)

                # Check if Already Exsists
                if candidate in self.trials_:
                    continue

                # Evaluate Candidate
                try:
                    self.eval_subset(candidate, X, y, groups)

                    if candidate.score > score:
                        score = candidate.score
                        subset = candidate

                except KeyboardInterrupt:
                    raise

                except:
                    pass

            # Stop Criteria
            if score < self.score_:
                continue

            # Update Subset
            self.subset_ = subset
            self.score_ = score

        return self

    def get_subset(self) -> Union[List[int], np.ndarray]:
        """
        Return the subset of selected features.

        Returns
        -------
        Union[List[int], np.ndarray]:
            A list or NumPy array of the indices of the selected features.
        """
        if (self.use_best is True) and hasattr(self, 'best_subset_'):
            # If using the best subset and the attribute `best_subset_` exists, return the best subset
            return self.best_subset_

        elif (self.use_best is False) and len(self.subset_) > 0:
            # If not using the best subset and the subset of selected features is not empty, return the last subset
            return self.last_subset_

        else:
            # If the model is not fitted or no subset is available, raise an error
            model_name = self.__class__.__name__
            raise NotFittedError(f'{model_name} is not fitted')


class GroupGreedSelector(_WrappedGroupSelector, GreedSelector):
    pass
