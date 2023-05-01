import abc
from time import time
from typing import List, Optional, Callable, Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from . import _plot_progress, _plot_subset
from . import FeatureSubset
from . import _print_last
from ..crossval import crossval


class _Selector(BaseEstimator, TransformerMixin):
    """
    A base class for feature selection transformers.

    Attributes
    ----------
    features_ : FeatureSubset
        The selected features to be used in `transform`.

    Methods
    -------
    transform(X: pd.DataFrame) -> pd.DataFrame:
        Reduce X to the selected features.
    get_subset() -> List[str]:
        Get list of columns to select.

    """

    def transform(self,
                  X: pd.DataFrame) -> pd.DataFrame:
        """Reduce X to the selected features.

        Parameters
        ----------
        X : DataFrame of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        Xt : DataFrame of shape [n_samples, n_selected_features]
            The input samples with only the selected features.

        """
        return X[self.get_subset()]

    @abc.abstractmethod
    def get_subset(self) -> FeatureSubset:
        """
        Get list of columns to select

        Returns
        -------
        use_cols : FeatureSubset
            Columns to select
        """
        return self.features_

    def _set_features(self,
                      X: pd.DataFrame) -> None:
        """
        Set the selected features based on the columns of `X`.

        Parameters
        ----------
        X : pd.DataFrame of shape [n_samples, n_features]
            The input samples.

        """
        self.features_ = FeatureSubset(X.columns)


class _WrappedSelector(_Selector):
    """
    Abstract base class for wrapped feature selectors.

    Parameters
    ----------
    estimator : estimator object
        The base estimator to fit on the feature subset.
    cv : int or iterable, optional (default=5)
        Determines the cross-validation splitting strategy.
    scoring : str, callable, or None, optional (default=None)
        A scoring metric to use for feature selection.
    max_iter : int, optional (default=20)
        The maximum number of iterations for the feature selector.
    max_time : float or None, optional (default=None)
        The maximum amount of time (in seconds) to run the feature selector.
    random_state : int, RandomState instance, or None, optional (default=0)
        Controls the random seed for the feature selector.
    n_jobs : int, optional (default=-1)
        The number of CPUs to use to run the feature selector.
    verbose : int, optional (default=1)
        Controls the verbosity of the feature selector.
    n_digits : int, optional (default=4)
        The number of digits to use for floating point output.
    cv_kwargs : dict, optional (default={})
        Additional keyword arguments to pass to the cross-validation splitter.

    Attributes
    ----------
    n_features_ : int
        The number of features in the dataset.
    min_features_ : int
        The minimum number of features to select.
    """

    @abc.abstractmethod
    def __init__(self,
                 estimator: BaseEstimator,
                 cv: int = 5,
                 scoring: Optional[Union[Callable, str]] = None,
                 max_iter: int = 20,
                 max_time: Optional[int] = None,
                 random_state: int = 0,
                 n_jobs: int = -1,
                 verbose: int = 1,
                 n_digits: int = 4,
                 cv_kwargs: Optional[dict] = None):

        if cv_kwargs is None:
            cv_kwargs = {}
        self.estimator = estimator
        self.scoring = scoring
        self.max_iter = max_iter
        self.max_time = max_time
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_digits = n_digits
        self.cv_kwargs = cv_kwargs

    @property
    def n_features_(self) -> int:
        """
        The number of features in the dataset.

        Returns
        -------
        n_features_ : int
            The number of features in the dataset.
        """
        return self.features_.n_features

    @property
    def min_features_(self) -> int:
        """
        The minimum number of features to select.

        Returns
        -------
        min_features_ : int
            The minimum number of features to select.

        """
        min_features = _check_k_features(self.min_features,
                                         self.n_features_,
                                         'min_features')
        return min_features

    @property
    def max_features_(self) -> int:
        """
        Gets the actual maximum number of features to select.
        """
        max_features = _check_k_features(self.max_features,
                                         self.n_features_,
                                         'max_features')
        msg = "<min_features> must be lower then <max_features>"
        assert self.min_features_ <= max_features, msg
        return max_features

    @staticmethod
    def _get_importance(subset: List[str],
                        result: Dict[str, Any]):
        """
        Extracts feature importance's from cross-validation results.

        Parameters:
        -----------
        subset : list of str
            The list of features in the current subset.

        result : dict
            The cross-validation results.

        Returns:
        --------
        subset : List[str]
            The input subset of features.
        """
        if 'importance' in result:
            imp = result['importance']
            subset.importance = pd.Series(np.average(imp, axis=0), index=subset)
            subset.importance_std = pd.Series(np.std(imp, axis=0), index=subset)
        return subset

    def _eval_subset(self,
                     subset: List[int],
                     X: pd.DataFrame,
                     y: pd.Series,
                     groups: np.ndarray) -> List[str]:
        """
        Evaluate the performance of a subset of features using cross-validation.

        Args:
            subset (List[int]): A list of feature indices to evaluate.
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target vector.
            groups (np.ndarray): The group vector.

        Returns:
            List[str]: The evaluated subset of features with updated score and importance.

        """
        result = crossval(estimator=self.estimator,
                          cv=self.cv,
                          X=X[subset],
                          y=y,
                          groups=groups,
                          scoring=self.scoring,
                          n_jobs=self.n_jobs,
                          return_pred=False,
                          verbose=0,
                          **self.cv_kwargs)

        subset.score = np.average(result['val_score'])
        subset.score_std = np.std(result['val_score'])
        subset = self._get_importance(subset, result)

        return subset

    def eval_subset(self,
                    subset: pd.Series,
                    X: pd.DataFrame,
                    y: pd.Series,
                    groups: Optional[pd.Series] = None):

        # Convert to FeatureSubset
        if type(subset) != type(self.features_):
            subset = self.features_.copy().set_subset(subset)

        # Evaluate
        tic = time()
        self._eval_subset(subset, X, y, groups)
        subset.eval_time = time() - tic

        # Update stats
        self.total_time_ = getattr(self, 'total_time_', .0) + subset.eval_time

        if not hasattr(self, 'best_score_') or self.best_score_ < subset.score:
            self.best_subset_ = subset
            self.best_score_ = subset.score

        # Update history
        subset.idx = self.n_iters_
        self.trials_.append(subset)

        # Verbose
        _print_last(self)

        # Check limits
        self._check_max_iter()
        self._check_max_time()

        return subset.score

    def _check_max_iter(self) -> None:
        """
        Checks if the maximum number of iterations has been reached and raises a KeyboardInterrupt if so.

        Returns
        -------
        Nothing :
            None
        """
        if hasattr(self, 'max_iter') and self.max_iter:
            if self.max_iter <= self.n_iters_:
                if self.verbose: print('Iterations limit exceed!')
                raise KeyboardInterrupt

    def _check_max_time(self) -> None:
        """
        Checks if the maximum time has been reached and raises a KeyboardInterrupt if so.

        Returns
        -------
        Nothing:
            None
        """
        if hasattr(self, 'max_time') and self.max_time:
            if self.max_time <= self.total_time_:
                if self.verbose: print('Time limit exceed!')
                raise KeyboardInterrupt

    def _reset_trials(self) -> None:
        """
        Resets the trials list to an empty list.

        Returns
        -------
        Nothing:
            None
        """
        self.trials_ = []

    @property
    def n_iters_(self):
        """
        Returns the number of trials that have been run so far.

        Returns
        -------

        Nothing:
            None
        """
        return len(self.trials_)

    # @property
    # def feature_importances_(self):
    #    subset = self._select_features()
    #    trial = _find_trial(subset)
    #    return pd.Series(trial['importance'], index=self.features_)

    # @property
    # def feature_importances_std_(self):
    #    subset = self._select_features()
    #    trial = _find_trial(subset)
    #    return pd.Series(trial['importance_std'], index=self.features_)

    def plot_progress(self,
                      **kwargs) -> tuple:
        """
        Plots the progress of the model during fitting.
        """
        return _plot_progress(self, **kwargs)

    def plot_subset(self,
                    **kwargs) -> tuple:
        """Plots the selected subset of features."""
        return _plot_subset(self, **kwargs)

    def get_subset(self) -> pd.Series:
        """
        Returns the best subset of features if the model has been fitted, otherwise raises a NotFittedError.
        """
        if hasattr(self, 'best_subset_'):
            return self.best_subset_
        else:
            model_name = self.__class__.__name__
            raise NotFittedError(f'{model_name} is not fitted')


def _check_k_features(k_features: Union[int, float],
                      n_features: int,
                      param: str = 'k_features') -> int:
    """
    Check if k_features is a valid value and return it as an integer.

    Parameters
    ----------
    k_features : int or float
        The value of k_features to check.
    n_features : int
        The number of features in the dataset.
    param : str, optional (default='k_features')
        The name of the parameter being checked.

    Returns
    -------
    int
        The value of k_features as an integer.

    Raises
    ------
    ValueError
        If k_features is not an int or float, or if it is not a valid value.

    """
    if isinstance(k_features, int):
        # If k_features is an integer, check if it is greater than 0.
        if k_features > 0:
            k_features = k_features
        else:
            raise ValueError(f'Integer <{param}> must be greater than 0')

    elif isinstance(k_features, float):
        # If k_features is a float, check if it is between 0 and 1 (exclusive).
        if 0 < k_features < 1:
            # Calculate the number of features to select based on the percentage
            # specified by k_features, and ensure that it is at least 1.
            k_features = max(k_features * n_features, 1)
            k_features = int(k_features)
        else:
            raise ValueError(f'Float <{param}> must be from interval (0, 1)')

    else:
        raise ValueError(f'Parameter <{param}> must be int or float,'
                         f'got {k_features}')

    return k_features


class _WrappedGroupSelector:
    """
    A class for wrapping a group selector algorithm and providing additional methods for handling feature groups.
    """

    def _get_importance(self,
                        subset: Any,
                        result: Dict[str, Any]) -> Any:
        """
        Computes feature importance for each group in the subset and sets the 'importance' and 'importance_std'
        attributes of the subset object.

        Parameters
        ----------
        subset : object
            A feature subset object.
        result : dict
            A dictionary containing the feature importance scores and their corresponding groups.

        Returns
        -------
        object:
            The updated feature subset object with the 'importance' and 'importance_std' attributes set.
        """
        # check if importance scores are available in the result
        if 'importance' in result:
            # extract the features and their corresponding importance scores
            features, imp = result['features'], result['importance']
            # extract the groups from the features
            groups = [group for group, _ in features]

            # create a dataframe with the importance scores and the corresponding groups
            imp = pd.DataFrame(imp, columns=groups).T
            # group the importance scores by their corresponding groups
            imp = imp.groupby(groups).sum()

            # compute the mean and standard deviation of the importance scores for each group
            subset.importance = imp.mean(axis=1)
            subset.importance_std = imp.std(axis=1)

        # return the updated feature subset object
        return subset

    def _set_features(self,
                      X: pd.DataFrame) -> None:
        """
        Sets the 'features_' attribute of the object as a FeatureSubset object containing the columns of the input data
        frame and their corresponding groups.

        Parameters
        ----------
        X : DataFrame
            The input data frame containing the features and their groups.
        """

        # create a FeatureSubset object with the columns of the input data frame and their corresponding groups
        self.features_ = FeatureSubset(X.columns, group=True)
