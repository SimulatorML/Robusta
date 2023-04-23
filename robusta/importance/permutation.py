from typing import Optional, Callable, List, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    is_classifier,
)
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.random import check_random_state
from tqdm import tqdm_notebook


def _get_col_score(estimator: BaseEstimator,
                   X: pd.DataFrame,
                   y: pd.Series,
                   col: str,
                   n_repeats: int,
                   scorer: Callable,
                   rstate: np.random.RandomState) -> np.ndarray:
    """Calculate score when `col` is permuted.

    Parameters:
    -----------
    estimator : BaseEstimator
        A scikit-learn estimator object implementing the fit and predict methods.
    X : pd.DataFrame
        Input data of shape (n_samples, n_features).
    y : pd.Series
        Target values of shape (n_samples,)
    col : str
        The name of the column to be permuted.
    n_repeats : int
        The number of times to permute `col` and calculate the score.
    scorer : callable
        A function that returns a scalar score based on `estimator`, `X`, and `y`.
    rstate : np.random.RandomState
        Random state to use for the permutation.

    Returns:
    --------
    scores : np.ndarray
        The scores obtained when permuting `col`. An array of shape (n_repeats,)
    """

    # Make a copy of the column to permute
    x = X.loc[:, col].copy()

    # Initialize array to store scores
    scores = np.zeros(n_repeats)

    for i in range(n_repeats):
        # Permute the column
        X.loc[:, col] = rstate.permutation(x)
        # Cast the column to the original dtype
        X.loc[:, col] = X.loc[:, col].astype(x.dtype)

        # Calculate the score with the estimator
        score = scorer(estimator, X, y)  # bottleneck
        scores[i] = score

    return scores


def get_col_score(estimator: BaseEstimator,
                  X: pd.DataFrame,
                  y: pd.Series,
                  col: str,
                  n_repeats: int = 5,
                  scoring: Optional[Union[str, Callable]] = None,
                  random_state: Optional[int] = None) -> np.ndarray:
    """
    Calculate the score when the column `col` of the input `X` is permuted.

    Parameters:
        estimator (BaseEstimator): The estimator to use for scoring.
        X (pd.DataFrame): The input data of shape (n_samples, n_features).
        y (pd.Series): The target data of shape (n_samples,).
        col (str): The name of the column to permute.
        n_repeats (int): The number of times to repeat the permutation. Default is 5.
        scoring (Optional[str, Callable]): The scoring metric to use. Default is None, which uses the estimator's default scorer.
        random_state (Optional[int]): The random seed to use. Default is None, which uses the global numpy random number generator.

    Returns:
        np.ndarray: The scores when `col` is permuted, of length `n_repeats`.
    """
    # Check the scoring metric
    scorer = check_scoring(estimator, scoring=scoring)

    # Check the random state
    rstate = check_random_state(random_state)

    # Get the scores when the column is permuted
    scores = _get_col_score(estimator=estimator,
                            X=X,
                            y=y,
                            col=col,
                            n_repeats=n_repeats,
                            scorer=scorer,
                            rstate=rstate)

    return scores


def _get_group_score(estimator: BaseEstimator,
                     X: pd.DataFrame,
                     y: pd.Series,
                     g: List[int],
                     n_repeats: int,
                     scorer: Callable,
                     rstate: np.random.RandomState) -> np.ndarray:
    """
    Calculate the score of an estimator on permuted data for a specified group of columns.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to use for calculating the score.
    X : pd.DataFrame
        The input data.
    y : pd.Series
        The target variable.
    g : List[int]
        The list of column indices to be permuted.
    n_repeats : int
        The number of times to permute the columns and calculate the score.
    scorer : Callable
        The scoring function to use for evaluating the estimator.
    rstate : np.random.RandomState
        The random state to use for generating the permutations.

    Returns
    -------
    scores : np.ndarray
        The array of scores obtained from permuting the columns.

    """

    # Create a copy of the dataframe X with only the columns in the list g
    x = X.loc[:, g].copy()

    # Create an empty numpy array of size n_repeats to store the scores
    scores = np.zeros(n_repeats)

    # For each permutation:
    for i in range(n_repeats):
        # Permute the columns in g using the random state rstate
        X.loc[:, g] = rstate.permutation(x)

        # Cast the permuted columns to the original data types
        X.loc[:, g] = X.loc[:, g].astype(x.dtypes)

        # Calculate the score of the estimator on the permuted data
        score = scorer(estimator, X, y)

        # Store the score in the scores array
        scores[i] = score

    # Return the scores array
    return scores


def get_group_score(estimator: BaseEstimator,
                    X: pd.DataFrame,
                    y: pd.Series,
                    g: List[int],
                    n_repeats: int = 5,
                    scoring: Optional[Union[str, Callable]] = None,
                    random_state: Optional[int] = None) -> np.ndarray:
    """
    Calculate the score of an estimator on permuted data for a specified group of columns.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to use for calculating the score.
    X : pd.DataFrame
        The input data.
    y : pd.Series
        The target variable.
    g : List[int]
        The list of column indices to be permuted.
    n_repeats : int, optional (default=5)
        The number of times to permute the columns and calculate the score.
    scoring : str, callable, or None, optional (default=None)
        The scoring method to use. If None, the estimator's default scorer is used.
        If a string, it must be a valid scoring metric for scikit-learn.
        If a callable, it must be a scoring function that takes an estimator, input data, and target variable as input.
    random_state : int or None, optional (default=None)
        Seed for the random number generator. If None, a random seed is used.

    Returns
    -------
    scores : np.ndarray
        The array of scores obtained from permuting the columns.

    """
    # Check the scoring method and random state
    scorer = check_scoring(estimator=estimator,
                           scoring=scoring)
    rstate = check_random_state(seed=random_state)

    # Calculate the scores using the _get_group_score function
    scores = _get_group_score(estimator=estimator,
                              X=X,
                              y=y,
                              g=g,
                              n_repeats=n_repeats,
                              scorer=scorer,
                              rstate=rstate)

    # Return the scores
    return scores


def permutation_importance(estimator: BaseEstimator,
                           X: pd.DataFrame,
                           y: pd.Series,
                           subset: Optional[list] = None,
                           scoring: Optional[Union[str, Callable]] = None,
                           n_repeats: int = 5,
                           n_jobs: Optional[int] = None,
                           random_state: int = 0,
                           tqdm: bool = False) -> dict:
    """
    Compute feature importance's using permutation importance.
    
    Parameters:
    estimator (BaseEstimator):
        A supervised learning estimator.
    X (pd.DataFrame):
        The input data for the estimator.
    y (pd.Series):
        The target variable for the estimator.
    subset (list, optional):
        A list of features to be considered for the analysis. If None, all the features in
        the dataframe X are considered. Default is None.
    scoring (str or callable, optional):
        The scoring function to use for feature importance. If None, the estimator
        default score function is used. Default is None.
    n_repeats (int, optional):
        The number of times to permute each feature. Default is 5.
    n_jobs (int, optional):
        The number of parallel jobs to run. If None, it uses one job per CPU. Default is None.
    random_state (int, optional):
        The random state to use. Default is 0.
    tqdm (bool, optional):
        Whether to use tqdm to display a progress bar. Default is False.

    Returns:
        A dictionary with the following keys:
            - importances_mean: The mean importance for each feature.
            - importances_std: The standard deviation of the importance for each feature.
            - importances: A 2D numpy array containing the importance scores for each feature in each repetition.
            - score: The estimator score on the original data.
    """
    # Get a list of all columns in the dataframe X
    columns = list(X.columns)

    # Check if the subset list contains only features from X
    msg = "<subset> must contain only features from <X>"
    assert not subset or not set(subset) - set(columns), msg

    # Check if the subset list contains only unique features
    msg = "<subset> must contain only unique features"
    assert not subset or len(set(subset)) == len(subset), msg

    # If subset is not given, use all the columns
    subset = subset if subset else columns

    # If tqdm flag is True, wrap subset with tqdm_notebook
    subset = tqdm_notebook(subset) if tqdm else subset

    # Check the scoring metric
    scorer = check_scoring(estimator=estimator,
                           scoring=scoring)

    # Create a random state object
    rstate = check_random_state(seed=random_state)

    # Get the base score of the estimator on the original data
    base_score = scorer(estimator, X, y)

    # FIXME: avoid <max_nbytes>
    # Use Parallel from joblib to compute the scores of each feature in the subset in parallel
    scores = Parallel(n_jobs=n_jobs,
                      max_nbytes='512M',
                      backend='multiprocessing')(
        delayed(_get_col_score)(estimator, X, y, feature, n_repeats, scorer, rstate)
        for feature in subset)

    # Create a numpy array to store the importance's for each feature
    importances = np.full((len(columns), n_repeats), np.nan)

    # Get the indices of each feature in the subset in the list of all columns
    ind = [columns.index(feature) for feature in subset]

    # Calculate the importances of each feature
    importances[ind] = base_score - np.array(scores)

    # Create a dictionary to store the results
    result = {'importances_mean': np.mean(importances, axis=1),
              'importances_std': np.std(importances, axis=1),
              'importances': importances,
              'score': base_score}

    # Return the results
    return result


def group_permutation_importance(estimator: BaseEstimator,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 subset: Optional[pd.Series] = None,
                                 scoring: Optional[Union[str, Callable]] = None,
                                 n_repeats: int = 5,
                                 n_jobs: Optional[int] = None,
                                 random_state: int = 0,
                                 tqdm: bool = False) -> dict:
    """
    Compute group permutation importance for a given estimator and dataset.
    
    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to evaluate the feature importance for.
    X : pd.DataFrame
        The data to compute the feature importances for.
    y : pd.Series
        The target variable.
    subset : Optional[pd.Series], default=None
        A list of feature group names to compute the importance for.
    scoring : Optional[str, Callable], default=None
        A scoring metric to use when evaluating the estimator's performance. If None, the default scoring metric for
        the estimator will be used.
    n_repeats : int, default=5
        The number of times to repeat the permutation of each group.
    n_jobs : Optional[int], default=None
        The number of CPU cores to use for parallel computation. If None, all available cores will be used.
    random_state : int, default=0
        The random state seed to use for the permutation.
    tqdm : bool, default=False
        Whether to display a progress bar during computation.

    Returns
    -------
    result : dict
        A dictionary containing the mean and standard deviation of the feature importances, the raw importance scores
        for each feature, and the base score of the estimator on the original data.
    """
    # Get a list of all top level columns in the dataframe X
    columns = list(X.columns.get_level_values(0).unique())

    # Check if the subset list contains only features from X
    msg = "<subset> must contain only features from <X>"
    assert not subset or not set(subset) - set(columns), msg

    # Check if the subset list contains only unique features
    msg = "<subset> must contain only unique features"
    assert not subset or len(set(subset)) == len(subset), msg

    # If subset is not given, use all the top level columns
    subset = subset if subset else columns

    # If subset is not given, use all the top level columns
    subset = tqdm_notebook(subset) if tqdm else subset

    # Check the scoring metric
    scorer = check_scoring(estimator=estimator,
                           scoring=scoring)

    # Create a random state object
    rstate = check_random_state(random_state)

    # Get the base score of the estimator on the original data
    base_score = scorer(estimator, X, y)

    # FIXME: avoid <max_nbytes>
    # Use Parallel from joblib to compute the scores of each group of features in the subset in parallel
    scores = Parallel(n_jobs=n_jobs,
                      max_nbytes='512M',
                      backend='multiprocessing')(
        delayed(_get_group_score)(estimator, X, y, feature, n_repeats, scorer, rstate)
        for feature in subset)

    # Create a numpy array to store the importance's for each feature
    importances = np.full((len(columns), n_repeats), np.nan)

    # Get the indices of each group of features in the subset in the list of all top level columns
    ind = [columns.index(feature) for feature in subset]

    # Calculate the importances of each group of features
    importances[ind] = base_score - np.array(scores)

    # Create a dictionary to store the results
    result = {'importances_mean': np.mean(importances, axis=1),
              'importances_std': np.std(importances, axis=1),
              'importances': importances,
              'score': base_score}

    # Return the results
    return result


class PermutationImportance(BaseEstimator, MetaEstimatorMixin):
    """
    A scikit-learn-compatible meta-estimator for computing feature importances
    using permutation importance.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator for which to compute feature importances.
    cv : str, default='prefit'
        Cross-validation strategy to use. If set to 'prefit', assumes that
        the estimator has already been fit.
    scoring : str or callable, default=None
        Scikit-learn scorer function or string identifier to use for scoring
        the estimator.
    n_repeats : int, default=5
        Number of times to repeat the permutation procedure.
    random_state : int or None, default=None
        Random seed to use for the permutation procedure.
    tqdm : bool, default=False
        Whether to display a progress bar.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
    y_transform : callable or None, default=None
        Function to transform the target variable before fitting the estimator.
    """

    def __init__(self,
                 estimator: BaseEstimator,
                 cv: str = 'prefit',
                 scoring: Optional[Union[str, Callable]] = None,
                 n_repeats: int = 5,
                 random_state: Optional[int] = None,
                 tqdm: bool = False,
                 n_jobs: Optional[int] = None,
                 y_transform: Optional[Callable] = None):
        self.feature_importances_std_ = None
        self.feature_importances_ = None
        self.scores_ = None
        self.raw_importances_ = None
        self.estimator = estimator
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.cv = cv
        self.random_state = random_state
        self.tqdm = tqdm
        self.n_jobs = n_jobs
        self.y_transform = y_transform

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            groups: Optional[pd.Series] = None,
            subset: Optional[pd.Series] = None,
            **fit_params) -> 'PermutationImportance':
        """
        Fits the estimator and computes feature importances using permutation
        importance.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.
        groups : np.array or None, default=None
            Group labels for samples, used in some cross-validation strategies.
        subset : list or None, default=None
            Subset of features to compute importances for. If None, all features
            are used.
        **fit_params : dict
            Additional parameters to pass to the estimator's `fit` method.

        Returns
        -------
        self : PermutationImportance
            The fitted estimator.
        """

        # check if cross-validation is specified or if there's only one training set
        if self.cv in ['prefit', None]:
            # create an array of indices for each row in X
            ii = np.arange(X.shape[0])

            # create a tuple with two identical arrays
            cv = np.array([(ii, ii)])

        else:
            cv = self.cv

        # check if the estimator is a classifier or not
        cv = check_cv(cv, y, classifier=is_classifier(self.estimator))

        # create an empty list to store the raw feature importances for each fold
        self.raw_importances_ = []

        # create an empty list to store the scores for each fold
        self.scores_ = []

        # iterate over each fold of the cross-validation
        for trn, oof in cv.split(X, y, groups):
            # get the training set for the current fold
            X_trn, y_trn = X.iloc[trn], y.iloc[trn]

            # get the validation set for the current fold
            X_oof, y_oof = X.iloc[oof], y.iloc[oof]

            # apply a transformation to y if provided
            y_trn_ = self.y_transform(y_trn) if self.y_transform else y_trn

            # if the estimator was already fit to data outside of this object
            if self.cv is 'prefit':
                # use the pre-fit estimator
                estimator = self.estimator

            else:
                # create a clone of the estimator and fit it to the training set for the current fold
                estimator = clone(self.estimator).fit(X_trn, y_trn_, **fit_params)

            # calculate the permutation importance for each feature using the validation set
            pi = permutation_importance(estimator=estimator,
                                        X=X_oof,
                                        y=y_oof,
                                        subset=subset,
                                        n_repeats=self.n_repeats,
                                        scoring=self.scoring,
                                        random_state=self.random_state,
                                        tqdm=self.tqdm,
                                        n_jobs=self.n_jobs)

            # create a DataFrame with the raw feature importances for the current fold
            imp = pd.DataFrame(pi['importances'], index=X.columns)

            # add the raw feature importances for the current fold to the list
            self.raw_importances_.append(imp)

            # add the score for the current fold to the list
            self.scores_.append(pi['score'])

        # concatenate the raw feature importances for each fold into a single DataFrame
        imps = pd.concat(self.raw_importances_, axis=1)

        # calculate the mean feature importance across folds
        self.feature_importances_ = imps.mean(axis=1)

        # calculate the standard deviation of feature importances across folds
        self.feature_importances_std_ = imps.std(axis=1)
        return self

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def score(self,
              X: pd.DataFrame,
              y: pd.Series = None,
              *args,
              **kwargs) -> float:
        """
        Returns the score of the wrapped estimator on the given data.

        Args:
            X (pd.DataFrame): Test input samples.
            y (pd.Series): True target values.
            **score_params: Additional score parameters.

        Returns:
            score (float): Score of the wrapped estimator on the given data.
        """
        return self.wrapped_estimator_.score(X, y, *args, **kwargs)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def predict(self,
                X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the class labels of the given data using the wrapped estimator.

        Args:
            X (pd.DataFrame): Input samples.

        Returns:
            y_pred (array-like of shape (n_samples,)): Predicted class labels.
        """
        return self.wrapped_estimator_.predict(X)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def predict_proba(self,
                      X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the probabilities of each class label for the given data using the wrapped estimator.

        Args:
            X (pd.DataFrame): Input samples.

        Returns:
            y_proba (array-like of shape (n_samples, n_classes)): Class probabilities.
        """
        return self.wrapped_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def predict_log_proba(self,
                          X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the logarithm of the probabilities of each class label for the given data using the wrapped estimator.

        Args:
            X (pd.DataFrame): Input samples.

        Returns:
            y_log_proba (array-like of shape (n_samples, n_classes)): Log of class probabilities.
        """
        return self.wrapped_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def decision_function(self,
                          X: pd.DataFrame) -> np.ndarray:
        """
        Computes the decision function of the given data using the wrapped estimator.

        Args:
            X (pd.DataFrame): Input samples.

        Returns:
            y_score (array-like of shape (n_samples, n_classes)): Decision function values.
        """
        return self.wrapped_estimator_.decision_function(X)

    @property
    def wrapped_estimator_(self) -> BaseEstimator:
        """
        Property that returns the wrapped estimator.

        If cv is "prefit" or refit is False, returns the original estimator.
        Otherwise, returns the estimator fitted on the entire dataset.

        Returns:
            wrapped_estimator_ (estimator object): Wrapped estimator.
        """
        if self.cv == "prefit" or not self.refit: return self.estimator
        return self.estimator_

    @property
    def _estimator_type(self) -> str:
        """
        Returns the type of the estimator.

        Returns:
            _estimator_type (str): Type of the estimator.
        """
        return self.estimator._estimator_type

    @property
    def classes_(self) -> np.ndarray:
        """
        Returns the classes of the wrapped estimator.

        Returns:
            classes_ (array-like of shape (n_classes,)): Class labels.
        """
        return self.wrapped_estimator_.classes_


class GroupPermutationImportance(PermutationImportance):
    """
    Compute group permutation feature importance for a given estimator.

    Parameters
    ----------
    estimator : BaseEstimator
        A supervised learning estimator that should have a ``fit`` method and a ``predict`` method.
    cv : str or cross-validation generator, default='prefit'
        Determines the cross-validation splitting strategy. Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation.
        - 'prefit', if the estimator is already fitted and ``transform`` or ``predict`` method can be directly
          applied to new data.
        - An iterable yielding (train, test) splits as arrays of indices. For each split, the estimator is cloned and
          fitted on the training data.
    scoring : str or callable, default=None
        Scoring metric to use for evaluating feature importance. If None, the estimator's default scoring metric
        is used.
    n_repeats : int, default=5
        Number of times to shuffle the data to estimate feature importances.
    random_state : int or None, default=None
        Seed of the pseudo random number generator used when shuffling the data.
    tqdm : bool, default=False
        Whether to display a progress bar when computing feature importances.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel. If None, then the number of jobs is set to the number of CPU cores.
    y_transform : callable, default=None
        A function to transform the target variable before passing it to the estimator's ``fit`` method.

    Attributes
    ----------
    feature_importances_std_ : pandas.Series
        The standard deviation of the feature importances across folds.
    feature_importances_ : pandas.Series
        The mean feature importances across folds.

    Methods
    -------
    fit(X, y, groups=None, subset=None, **fit_params)
        Fit the estimator and compute group permutation feature importances for each feature group.

    """
    def __init__(self,
                 estimator: BaseEstimator,
                 cv: str = 'prefit',
                 scoring: Optional[Union[str, Callable]] = None,
                 n_repeats: int = 5,
                 random_state: Optional[int] = None,
                 tqdm: bool = False,
                 n_jobs: Optional[int] = None,
                 y_transform: Optional[Callable] = None):
        super().__init__(estimator, cv, scoring, n_repeats, random_state, tqdm, n_jobs, y_transform)
        self.feature_importances_std_ = None
        self.feature_importances_ = None

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            groups: Optional[pd.Series] = None,
            subset: Optional[pd.Series] = None,
            **fit_params) -> 'GroupPermutationImportance':
        """
        Compute group permutation importance for feature selection.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        y : pd.Series
            Target variable.
        groups : pd.Series, optional
            Groups that features belong to.
        subset : pd.Series, optional
            Subset of features to evaluate for importance. If None, all features are used.
        **fit_params : dict
            Additional keyword arguments to pass to the estimator's `fit` method.

        Returns
        -------
        self : GroupPermutationImportance
            Fitted instance of the class.
        """
        # Extract unique column groups from X
        col_group = X.columns.get_level_values(0).unique()

        # If cv is prefit or None, create one fold for cross-validation
        if self.cv in ['prefit', None]:
            ii = np.arange(X.shape[0])
            cv = np.array([(ii, ii)])
        else:
            cv = self.cv

        # Check if cv is a valid cross-validation object
        cv = check_cv(cv=cv,
                      y=y,
                      classifier=is_classifier(self.estimator))

        # Initialize empty lists for storing raw importances and scores
        self.raw_importances_ = []
        self.scores_ = []

        # Extract unique column groups from X
        col_group = X.columns.get_level_values(0).unique()

        # If cv is prefit or None, create one fold for cross-validation
        if self.cv in ['prefit', None]:
            ii = np.arange(X.shape[0])
            cv = np.array([(ii, ii)])
        else:
            cv = self.cv

        # Check if cv is a valid cross-validation object
        cv = check_cv(cv=cv,
                      y=y,
                      classifier=is_classifier(self.estimator))

        # Initialize empty lists for storing raw importances and scores
        self.raw_importances_ = []
        self.scores_ = []

        # Loop through each fold of the cross-validation
        for trn, oof in cv.split(X, y, groups):
            # Get training and out-of-fold data
            X_trn, y_trn = X.iloc[trn], y.iloc[trn]
            X_oof, y_oof = X.iloc[oof], y.iloc[oof]

            # Apply y_transform function to y_trn if it's specified
            y_trn_ = self.y_transform(y_trn) if self.y_transform else y_trn

            # If cv is prefit, use the estimator as it is, otherwise clone it and fit to training data
            if self.cv is 'prefit':
                estimator = self.estimator
            else:
                estimator = clone(self.estimator).fit(X_trn, y_trn_, **fit_params)

            # Calculate group permutation importance on out-of-fold data
            pi = group_permutation_importance(estimator=estimator,
                                              X=X_oof,
                                              y=y_oof,
                                              subset=subset,
                                              n_repeats=self.n_repeats,
                                              scoring=self.scoring,
                                              random_state=self.random_state,
                                              tqdm=self.tqdm,
                                              n_jobs=self.n_jobs)

            # Save the raw feature importances to a list
            imp = pd.DataFrame(pi['importances'], index=col_group)
            self.raw_importances_.append(imp)

            # Save the score of the current fold to a list
            self.scores_.append(pi['score'])

        # Combine the raw feature importances into one dataframe and calculate the mean and std over the folds
        imps = pd.concat(self.raw_importances_, axis=1)
        self.feature_importances_ = imps.mean(axis=1)
        self.feature_importances_std_ = imps.std(axis=1)

        # Return the fitted instance of the class
        return self
