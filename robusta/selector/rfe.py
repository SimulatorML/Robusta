from typing import Optional, Callable, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from . import _WrappedSelector, _WrappedGroupSelector
from ..importance import (
    PermutationImportance,
    GroupPermutationImportance,
    ShuffleTargetImportance,
)


class RFE(_WrappedSelector):
    """
    Feature ranking with recursive feature elimination (RFE) and
    cross-validated selection of the best number of features.

    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of recursive feature elimination
    (RFE) is to select features by recursively considering smaller and smaller
    sets of features. First, the estimator is trained on the initial set of
    features and the importance of each feature is obtained either through a
    <coef_> attribute or through a <feature_importances_> attribute. Then, the
    least important features are pruned from current set of features. That
    procedure is recursively repeated on the pruned set until the desired
    number of features to select is eventually reached.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        The estimator must have either a <feature_importances_> or <coef_>
        attribute after fitting.

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

    min_features : int or float, optional (default=0.5)
        The number of features to select. Float values means percentage of
        features to select. E.g. value 0.5 (by default) means 50% of features.

    step : int or float, optional (default=1)
        The number of features to remove on each step. Float values means
        percentage of left features. E.g. 0.2 means 20% reduction on each step:
        500 => 400 => 320 => ...

    random_state : int or None (default=0)
        Random seed for permutations in PermutationImportance.
        Ignored if <importance_type> set to 'inbuilt'.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int, optional (default=1)
        Verbosity level

    Attributes
    ----------
    use_cols_ : list of string
        Feature names to select

    n_features_ : int
        Number of selected features

    min_features_ : int
        Minimum number of features

    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cv: int = 5,
        scoring: Optional[Union[str, Callable]] = None,
        min_features: float = 0.5,
        step: int = 1,
        use_best: bool = True,
        n_jobs: Optional[int] = None,
        verbose: int = 1,
        n_digits: int = 4,
        cv_kwargs: Optional[dict] = None,
    ):
        if cv_kwargs is None:
            cv_kwargs = {}
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.cv_kwargs = cv_kwargs

        self.min_features = min_features
        self.step = step
        self.use_best = use_best

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_digits = n_digits

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None
    ) -> "RFE":
        """
        Fit the RFE model to the data.

        Parameters
        ----------
        X : DataFrame
            The data.
        y : Series
            The target variable.
        groups : Optional[pd.Series], optional
            A vector indicating which observations belong to which groups.

        Returns
        -------
        self : object
            RFE: The fitted RFE model.
        """

        # Prepare data and set up selector
        self._fit_start(X)

        # Fit selector to data
        self._fit(X, y, groups)

        return self

    def partial_fit(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None
    ) -> "RFE":
        """
        Fit the RFE model to a subset of the data.

        This function is used when the data is too large to fit into memory all at once.

        Parameters
        ----------
        X : DataFrame
            A subset of the data.
        y : Series
            A subset of the variable.
        groups : Optional[pd.Series], optional
            A vector indicating which observations belong to which groups.

        Returns
        -------
        self : object
            RFE: The partially fitted RFE model.
        """

        # Prepare data and set up selector
        self._fit_start(X, partial=True)

        # Partially fit selector to data
        self._fit(X, y, groups)

        return self

    def _fit_start(self, X: pd.DataFrame, partial: bool = False) -> "RFE":
        """
        Fits the RFE model to the given dataset in a partial manner.

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
        self : RFE
            The fitted RFE model.
        """
        if not partial:
            self._set_features(X)
            self.subset_ = self.features_

            self._reset_trials()

        self.k_range_ = []
        k_features = self.n_features_

        # Loop until the minimum number of features is reached
        while k_features > self.min_features_:
            # Check the step
            step = _check_step(self.step, k_features, self.min_features_)

            # Calculate the new number of features
            k_features = k_features - step

            # Append to the list of feature numbers to be considered
            self.k_range_.append(k_features)

        # Calculate the maximum number of iterations
        self.max_iter = len(self.k_range_) + getattr(self, "n_iters_", 0) + 1

        # Create an iterator over the feature numbers to be considered
        self.k_range_ = iter(self.k_range_)

        return self

    @property
    def k_features_(self) -> int:
        """
        Return the number of features in the current subset.

        Returns
        -------
        num_features : int
            The number of features in the current subset.

        """
        return len(self.subset_)

    @property
    def forward(self) -> bool:
        """
        Return whether the RFE model is running in forward mode.

        Returns
        -------
        flag : bool
            Whether the RFE model is running in forward mode.

        """
        return False

    def _fit(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> "RFE":
        """
        Perform the recursive feature elimination algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The feature matrix to fit.

        y : pd.Series
            The target variable.

        groups : pd.Series
            Group labels for the samples. If provided, the evaluation will be performed
            by group.

        Returns
        -------
        self : RFE
            The fitted RFE instance.
        """

        # Evaluate the importance of the current feature subset
        self.eval_subset(self.subset_, X, y, groups)

        # Iterate over the feature subset sizes in descending order
        for k in self.k_range_:
            try:
                # Get the feature importance scores of the current subset
                scores = self.subset_.importance

                # Select the k best features according to the scores
                subset = _select_k_best(scores, k)

                # Store the current subset as the child of the previous subset
                parent = self.subset_
                self.subset_ = self.subset_.copy().set_subset(subset)
                self.subset_.parents = [parent]

                # Evaluate the performance of the new subset
                self.eval_subset(self.subset_, X, y, groups)

                # Stop iterating if the minimum number of features has been reached
                if self.k_features_ <= self.min_features_:
                    break

            except KeyboardInterrupt:
                break

        return self

    def get_subset(self):
        """
        Get the feature subset that was selected by the RFE algorithm.

        Returns
        -------
        subset : RFESubset
            The feature subset that was selected by the RFE algorithm.

        Raises
        ------
        NotFittedError
            If the RFE model has not been fitted yet.
        """
        if (self.use_best is True) and self.n_iters_ > 0:
            return self.best_subset_

        elif (self.use_best is False) and len(self.subset_) > 0:
            return self.subset_

        else:
            model_name = self.__class__.__name__
            raise NotFittedError(f"{model_name} is not fitted")


class GroupRFE(_WrappedGroupSelector, RFE):
    pass


class PermutationRFE(RFE):
    """
    Recursive feature elimination with permutation feature importance as a scoring metric.

    This class is a modification of sklearn's Recursive Feature Elimination (RFE) class that uses permutation
    feature importance as a scoring metric for ranking and selecting features.

    Parameters
    ----------
    estimator : BaseEstimator
        A supervised learning estimator with a fit method that provides information about feature importance.
    cv : int, default=5
        Determines the cross-validation splitting strategy.
    scoring : str or callable, default=None
        Scoring metric to use for feature ranking. If None, estimator's default scoring method is used.
    min_features : float, default=0.5
        Minimum number of features to select.
    step : int, default=1
        Number of features to remove at each iteration.
    n_repeats : int, default=5
        Number of times to shuffle the target variable when calculating permutation feature importance.
    random_state : int, default=0
        Random state for reproducibility.
    use_best : bool, default=True
        Whether to use the best subset of features determined by RFE or use all the features selected by RFE.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel for permutation feature importance. None means using all processors.
    verbose : int, default=1
        Controls verbosity of output during RFE.
    n_digits : int, default=4
        Number of decimal places to round the score and importance values.
    tqdm : bool or None, default=None
        Whether to show progress bar during permutation feature importance. If None, show progress bar if verbose > 0.
    y_transform : callable or None, default=None
        A function to transform the target variable before computing permutation feature importance. If None,
        the original target variable is used.

    Attributes
    ----------
    n_features_ : int
        The number of selected features.
    support_ : ndarray of shape (n_features,)
        The mask of selected features.
    ranking_ : ndarray of shape (n_features,)
        The feature ranking, such that `ranking_[i]` corresponds to the ranking position of the i-th feature.
    scores_ : ndarray of shape (n_features,)
        The cross-validation scores of the selected features.
    score_ : float
        The cross-validation score of the selected features.
    importances_ : ndarray of shape (n_features,)
        The mean permutation feature importance of the selected features.
    importances_std_ : ndarray of shape (n_features,)
        The standard deviation of the permutation feature importance of the selected features.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cv: Union[int, Callable] = 5,
        scoring: Optional[Union[str, Callable]] = None,
        min_features: float = 0.5,
        step: int = 1,
        n_repeats: int = 5,
        random_state: int = 0,
        use_best: bool = True,
        n_jobs: Optional[int] = None,
        verbose: int = 1,
        n_digits: int = 4,
        tqdm: Optional[bool] = None,
        y_transform: Optional[Callable] = None,
    ):
        super().__init__(
            estimator,
            cv,
            scoring,
            min_features,
            step,
            use_best,
            n_jobs,
            verbose,
            n_digits,
        )
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.y_transform = y_transform

        self.min_features = min_features
        self.step = step

        self.n_repeats = n_repeats
        self.random_state = random_state
        self.use_best = use_best

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_digits = n_digits
        self.tqdm = tqdm

    def _eval_subset(
        self,
        subset: pd.Series,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Evaluates the subset of features using permutation feature importance as a scoring metric.

        Parameters
        ----------
        subset : pd.Series
            A boolean mask indicating the features to evaluate.
        X : pd.DataFrame
            The input data with shape `(n_samples, n_features)`.
        y : pd.Series
            The target variable with shape `(n_samples,)`.
        groups : pd.Series, default=None
            Group labels for the samples used in cross-validation.

        Returns
        -------
        pd.Series
            A Series object containing the evaluation results for the given subset of features.
        """

        # Create a PermutationImportance object with the same settings as this class
        perm = PermutationImportance(
            estimator=self.estimator,
            cv=self.cv,
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            tqdm=self.tqdm,
            y_transform=self.y_transform,
        )

        perm.fit(X[subset], y, groups)

        # Calculate the mean score and standard deviation of the score
        subset.score = np.average(perm.scores_)
        subset.score_std = np.std(perm.scores_)

        # Extract the feature importance values and standard deviation from the permutation object
        subset.importance = perm.feature_importances_
        subset.importance_std = perm.feature_importances_std_

        return subset


class GroupPermutationRFE(_WrappedGroupSelector, PermutationRFE):
    """
    Recursive feature elimination with cross-validation based on permutation importance
    for grouped features. This class extends the PermutationRFE class from scikit-learn.
    It removes features recursively, based on the importance score provided by
    GroupPermutationImportance, a class that computes the permutation importance of
    each feature with respect to the target variable, considering group structure.

    Parameters
    ----------
    estimator : estimator object
        A supervised learning estimator that has a fit method.
    cv : int or cross-validation generator
        Determines the cross-validation splitting strategy.
        If int, it specifies the number of folds in a KFold.
    scoring : str, callable or None, optional, default=None
        Scoring metric to use for feature selection. If None, the estimator's default
        scorer is used.
    n_repeats : int, optional, default=5
        Number of times to permute each feature.
    n_jobs : int or None, optional, default=None
        The number of CPUs to use to do the computation. None means '1' unless in a
        joblib.parallel_backend context. -1 means using all processors.
    random_state : int or RandomState, optional, default=None
        Pseudo-random number generator state used for random sampling. Pass an int
        for reproducible results across multiple function calls.
    tqdm : callable, optional, default=None
        Progress bar function to use. If None, no progress bar will be displayed.

    Attributes
    ----------
    estimator_ : estimator object
        The estimator used to fit the data.
    n_features_ : int
        The number of selected features.
    support_ : array of shape [n_features]
        The mask of selected features.
    ranking_ : array of shape [n_features]
        The feature ranking, such that ranking_[i] corresponds to the ranking
        position of the i-th feature. Selected (i.e., estimated best) features
        are assigned rank 1.
    scores_ : array of shape [n_features]
        The scores of the features during the RFE.
    scores_std_ : array of shape [n_features]
        The std deviation of the scores of the features during the RFE.
    importances_ : array of shape [n_features, n_repeats]
        The feature importances during the RFE.
    importances_std_ : array of shape [n_features, n_repeats]
        The std deviation of the feature importances during the RFE.
    """

    def _eval_subset(
        self,
        subset: pd.Series,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Evaluate a subset of features using GroupPermutationImportance to compute
        the permutation importance of each feature with respect to the target variable,
        considering group structure. This method is called by the fit method in the
        PermutationRFE class.

        Parameters
        ----------
        subset : pandas Series
            A boolean mask that indicates the subset of features to evaluate.
        X : pandas DataFrame of shape (n_samples, n_features)
            The input data.
        y : pandas Series of shape (n_samples,)
            The target values.
        groups : pandas Series of shape (n_samples,), optional

        Returns
        -------
        subset : pandas Series
            The input subset of features, with the following attributes updated:
            - score: float, the average score obtained by the subset
            - score_std: float, the standard deviation of the scores obtained by the subset
            - importance: ndarray of shape (n_features,), the average feature importance
                          scores obtained by the subset
            - importance_std: ndarray of shape (n_features,), the standard deviation of the
                              feature importance scores obtained by the subset

        """

        # Instantiate a GroupPermutationImportance object with the provided parameters
        perm = GroupPermutationImportance(
            estimator=self.estimator,
            cv=self.cv,
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            tqdm=self.tqdm,
        )

        # Compute the permutation importance of the features in the subset
        perm.fit(X[subset], y, groups)

        # Update the subset Series with the scores and feature importances
        subset.score = np.average(perm.scores_)
        subset.score_std = np.std(perm.scores_)

        subset.importance = perm.feature_importances_
        subset.importance_std = perm.feature_importances_std_

        return subset


class ShuffleRFE(RFE):
    """
    Recursive feature elimination with shuffling of the target variable.

    Parameters
    ----------
    estimator : scikit-learn estimator
        The base estimator used for feature selection.

    cv : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation
        - integer, to specify the number of folds in a `(Stratified)KFold`
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

    scoring : string or callable, default=None
        A string (see scikit-learn's `scoring` parameter) or callable to evaluate
        the predictions on the validation set. If None, the estimator's default scorer
        is used.

    min_features : float, default=0.5
        The minimum percentage of features to retain at each iteration.
        Must be between 0 and 1.

    step : int, default=1
        The number of features to remove at each iteration.

    n_repeats : int, default=5
        The number of times to repeat the permutation or shuffle procedure.

    random_state : int, default=0
        Controls the random seed for the permutation or shuffle procedure.

    use_best : bool, default=True
        Whether to use the best set of features as returned by the feature selector
        after fitting.

    n_jobs : int or None, default=None
        The number of jobs to run in parallel. None means using one core.
        Only used when `tqdm=True`.

    tqdm : bool, default=False
        Whether to show the progress bar during feature selection.

    verbose : int, default=0
        Controls the verbosity of the feature selector.

    cv_kwargs : dict or None, default=None
        Additional arguments to be passed to the cross-validator, if any.

    Attributes
    ----------
    n_features_ : int
        The number of selected features.

    support_ : ndarray of shape (n_features,)
        The mask of selected features.

    ranking_ : ndarray of shape (n_features,)
        The feature ranking. The larger the number, the lower the importance.

    subsets_ : list of pandas Series
        The subsets of features selected at each iteration. Each subset has
        the following attributes:
        - score: float, the average score obtained by the subset
        - score_std: float, the standard deviation of the scores obtained by the subset
        - importance: ndarray of shape (n_features,), the average feature importance
                      scores obtained by the subset
        - importance_std: ndarray of shape (n_features,), the standard deviation of the
                          feature importance scores obtained by the subset
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cv: int = 5,
        scoring: Optional[Union[str, Callable]] = None,
        min_features: float = 0.5,
        step: int = 1,
        n_repeats: int = 5,
        gain: str = "dif",
        random_state: int = 0,
        use_best: bool = True,
        n_jobs: Optional[int] = None,
        tqdm: bool = False,
        verbose: int = 0,
        cv_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            estimator,
            cv,
            scoring,
            min_features,
            step,
            use_best,
            n_jobs,
            verbose,
            cv_kwargs,
        )
        if cv_kwargs is None:
            cv_kwargs = {}
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.cv_kwargs = cv_kwargs

        self.min_features = min_features
        self.step = step

        self.n_repeats = n_repeats
        self.gain = gain
        self.random_state = random_state
        self.use_best = use_best

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.tqdm = tqdm

    def _eval_subset(
        self,
        subset: pd.Series,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Evaluate a subset of features using ShuffleTargetImportance.

        Parameters
        ----------
        subset : Series
            The subset of features to evaluate.
        X : DataFrame
            The training data.
        y : Series
            The target variable.
        groups : Optional[pd.Series], optional
            The group labels for group-wise shuffle. Defaults to None.

        Returns
        -------
        result : pd.Series
            A pandas Series object with the evaluation results.
        """

        # Initialize a ShuffleTargetImportance object
        shuff = ShuffleTargetImportance(
            estimator=self.estimator,
            cv=self.cv,
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            n_jobs=self.n_jobs,
            tqdm=self.tqdm,
            random_state=self.random_state,
            cv_kwargs=self.cv_kwargs,
        )

        # Fit the ShuffleTargetImportance object on the subset of features
        shuff.fit(X[subset], y, groups)

        # Compute the mean and standard deviation of the scores obtained
        # by ShuffleTargetImportance
        subset.score = np.average(shuff.scores_)
        subset.score_std = np.std(shuff.scores_)

        # Store the feature importances obtained by ShuffleTargetImportance
        subset.importance = shuff.feature_importances_
        subset.importance_std = shuff.feature_importances_std_

        return subset


class GroupShuffleRFE(_WrappedGroupSelector, ShuffleRFE):
    """
    GroupShuffleRFE is a class for recursive feature elimination with group k-fold cross-validation and
    shuffled feature importance. It is a child class of _WrappedGroupSelector and ShuffleRFE.

    Parameters:
    -----------
    estimator: object
        A supervised learning estimator with a fit method that provides information about feature importance
    cv: int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy
    scoring: string, callable or None, optional, default=None
        A scoring metric to evaluate feature importance
    n_repeats: int, optional, default=5
        Number of times cross-validation will be repeated for each feature subset
    gain: string, optional, default='mean'
        The gain metric to determine feature importance
    n_jobs: int, optional, default=-1
        The number of CPUs to use to do the computation
    tqdm: callable or None, optional, default=None
        A function to show the progress of the calculation
    random_state: int or RandomState instance, optional, default=None
        Determines the random number generation
    cv_kwargs: dict, optional, default=None
        Optional parameters to be passed to the cross-validation strategy

    Attributes:
    -----------
    scores_: list of float
        The average scores for each subset
    importances_: list of numpy.ndarray
        The importances for each subset
    support_: numpy.ndarray
        The mask of selected features

    """

    def _eval_subset(
        self,
        subset: pd.Series,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Evaluate a subset of features by computing their importance using ShuffleTargetImportance
        with group-aware cross-validation.

        Parameters
        ----------
        subset:
            A pandas Series representing a subset of features to evaluate.
        X:
            A pandas DataFrame representing the input features.
        y:
            A pandas Series representing the target variable.
        groups:
            A pandas Series representing the groups for group-aware cross-validation.

        Returns
        -------
        subset:
            A pandas Series containing the evaluated subset of features and their importance scores.
        """
        shuff = ShuffleTargetImportance(
            estimator=self.estimator,
            cd=self.cv,
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            n_jobs=self.n_jobs,
            tqdm=self.tqdm,
            random_state=self.random_state,
            cv_kwargs=self.cv_kwargs,
        )
        shuff.fit(X[subset], y, groups)

        # Compute mean and standard deviation of the score
        subset.score = np.average(shuff.scores_)
        subset.score_std = np.std(shuff.scores_)

        # Get the raw feature importances and the feature names
        result = dict(importance=shuff.raw_importances_, features=list(X[subset]))

        # Compute the feature importance using the _get_importance method
        subset = self._get_importance(subset, result)

        return subset


def _select_k_best(scores: pd.Series, k_best: int) -> pd.Index:
    """
    Selects the k_best best features based on their scores.

    Parameters
    ----------
    scores : Series
        A Pandas Series containing the feature scores.
    k_best : int
        The number of best features to select.

    Returns
    -------
    best_features:
        A list of the indices of the k_best best features.
    """
    return scores.index[np.argsort(-scores.values)][:k_best]


def _check_step(step: Union[int, float], n_features: int, k_features: int) -> int:
    """
    Check the validity of the <step> parameter and convert it to integer.

    Parameters
    ----------
    step : Union[int, float]
        The step size of the number of features to remove at each iteration.
    n_features : int
        The total number of features in the dataset.
    k_features : int
        The desired number of features to select.

    Returns
    -------
    int:
        The validated integer value of the <step> parameter.

    Raises:
    ValueError:
        If the <step> parameter is not an int or float.
    ValueError:
        If the <step> parameter is an integer that is less than or equal to 0.
    ValueError:
        If the <step> parameter is a float that is not in the interval (0, 1).
    """

    # Check if <step> is an integer
    if isinstance(step, int):
        # If <step> is positive, keep it as is
        if step > 0:
            step = step

        # Otherwise, raise an error
        else:
            raise ValueError("Integer <step> must be greater than 0")

    # Check if <step> is a float
    elif isinstance(step, float):
        # If <step> is within the range (0, 1), calculate the number of features to remove and convert it to an integer
        if 0 < step < 1:
            step = max(step * n_features, 1)
            step = int(step)
        # Otherwise, raise an error
        else:
            raise ValueError("Float <step> must be from interval (0, 1)")

    # If <step> is neither an integer nor a float, raise an error
    else:
        raise ValueError(f"Parameter <step> must be int or float, got {step}")

    # Return the minimum of <step> and the difference between the total number of features and the
    # desired number of features
    return min(step, int(n_features - k_features))
