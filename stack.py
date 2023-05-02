from typing import List, Optional, Callable, Union, Dict, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import (
    clone,
    is_classifier,
    TransformerMixin,
    RegressorMixin,
    ClassifierMixin,
    BaseEstimator,
    MetaEstimatorMixin,
)
from sklearn.model_selection import check_cv
from sklearn.utils.metaestimators import _BaseComposition

from .crossval import _predict, _check_avg, _avg_preds
from .crossval import crossval


class StackingTransformer(_BaseComposition, TransformerMixin):
    """
    Stacking Transformer with inbuilt Cross-Validation

    Parameters
    ----------
    estimators : list
        List of (name, estimator) tuples (implementing fit/predict).

    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy:

        - None, use non-CV predictions
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An iterable yielding (train, test) splits as arrays of indices.

    scoring : string, callable or None, optional, default: None
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only a single value.
        If None, the estimator's default scorer (if available) is used.

    test_avg : bool (default=True)
        Stacking strategy (essential parameter).

        See robusta.crossval for details.

    avg_type : string, {'soft', 'hard', 'auto', 'rank'} (default='auto')
        Averaging strategy for aggregating different CV folds predictions.

        See 'crossval' from 'robusta.crossval' for details.

    method : {'predict', 'predict_proba'}, optional (defaul='predict')
        Invokes the passed method name of the passed estimators.

    join_X : bool (default=False)
        If True, concatenate stacked predictions with the original data

    n_jobs : int or None, optional (default=-1)
        Number of jobs to run in parallel. None means 1.

    verbose : int (default=1)
        Verbosity level

    n_digits : int (default=4)
        Verbose score precision

    """

    def __init__(
        self,
        estimators: List[BaseEstimator],
        cv: int = 5,
        scoring: Optional[Union[str, Callable]] = None,
        test_avg: bool = True,
        avg_type: str = "auto",
        method: str = "predict",
        join_X: bool = False,
        n_jobs: int = -1,
        verbose: int = 0,
        n_digits: int = 4,
        random_state: int = 0,
    ):
        self.estimators = estimators
        self.cv = cv
        self.scoring = scoring

        self.test_avg = test_avg
        self.avg_type = avg_type
        self.method = method
        self.join_X = join_X

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_digits = n_digits
        self.random_state = random_state

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Check if the data is in train or test mode
        if self._is_train(X):
            # If in train mode, transform the data with the first layer estimators
            S = self._transform_train(X)

        else:
            # If in test mode, transform the data with the trained first layer estimators
            S = self._transform(X)

        # Concatenate the transformed data with the original data if join_X is True
        if self.join_X:
            return X.join(S)
        else:
            return S

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None
    ) -> "StackingTransformer":
        """
        Fits the stacked transformer using the input training data and labels.

        Parameters
        ----------
        X : DataFrame
            Input training data.
        y : Series
            Training labels.
        groups : pandas.Series, optional
            Group labels for cross-validation. Default is None.

        Returns
        -------
        StackingTransformer:
            The fitted stacked transformer object.
        """

        # Check the type and name of each estimator in the list
        _check_estimator_types(self.estimators)
        _check_estimator_names(self.estimators)

        self._save_train(X, y)
        self._fit_1st_layer(X, y, groups)

        return self

    def _fit_1st_layer(
        self, X: pd.DataFrame, y: pd.Series, groups: pd.Series
    ) -> "StackingTransformer":
        """
        Fits the first layer estimators using cross-validation and saves the results.

        Parameters
        ----------
        X : DataFrame
            Input training data.
        y : Series
            Training labels.
        groups : Series
            Group labels for cross-validation.

        Returns
        -------
        StackingTransformer:
            The fitted stacked transformer object.
        """

        # Get the names of the first layer estimators
        self.names_ = [name for name, _ in self.estimators]

        # Initialize instance variables
        self.estimators_A_ = []
        self.estimators_B_ = []
        self.scores_std_ = []
        self.scores_ = []

        # Perform cross-validation
        cv = check_cv(self.cv, y, is_classifier(self.estimators[0]))
        self.folds_ = list(cv.split(X, y, groups))

        for name, estimator in self.estimators:
            # Fit the estimator using cross-validation
            result = crossval(
                estimator=estimator,
                cv=self.folds_,
                X=X,
                y=y,
                groups=groups,
                X_new=None,
                test_avg=self.test_avg,
                avg_type=self.avg_type,
                scoring=self.scoring,
                method=self.method,
                verbose=self.verbose,
                n_digits=self.n_digits,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                return_estimator=True,
            )

            estimators = result["estimator"]
            if self.test_avg:
                self.estimators_A_.append(estimators)
                self.estimators_B_ = None
            else:
                self.estimators_A_.append(estimators[:-1])
                self.estimators_B_.append(estimators[-1:])

            scores = result["score"]
            self.scores_.append(np.mean(scores))
            self.scores_std_.append(np.std(scores))

        return self

    def _save_train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Saves the training data and labels for later use.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas DataFrame containing the training data.
        y : pd.Series
            A pandas Series containing the training labels.

        Returns
        -------
        Nothing:
            None
        """
        self._train_shape = X.shape
        self._train_index = X.index
        self._y = y.copy()

    def _is_train(self, X: pd.DataFrame) -> bool:
        """
        Checks if the input DataFrame is the same as the training data.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas DataFrame to be checked.

        Returns
        -------
        boolean:
            A boolean indicating whether the input DataFrame is the same as the training data.
        """
        if (X.shape is self._train_shape) and (X.index is self._train_index):
            return True
        else:
            return False

    def _transform_train(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the training data using the first layer estimators.

        Parameters
        ----------
        X : DataFrame
            A pandas DataFrame containing the training data.

        Returns
        -------
        DataFrame:
            A pandas DataFrame containing the transformed training data.
        """

        # Loop over each set of first layer estimators
        pred_list = []
        for estimators in self.estimators_A_:
            # Check the average and method used by the first estimator
            avg, method = _check_avg(estimators[0], self.avg_type, self.method)

            # Make predictions on the out-of-fold (oof) samples in parallel
            preds = Parallel(n_jobs=self.n_jobs)(
                (
                    delayed(_predict)(estimator, method, X.iloc[oof], self._y)
                    for estimator, (trn, oof) in zip(estimators, self.folds_)
                )
            )

            # Average the predictions and append them to the list
            pred = _avg_preds(preds, avg, X, self._y)
            pred_list.append(pred)

        # Stack the predictions from each set of first layer estimators
        S = stack_preds(pred_list, self.names_)
        return S

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the test data using the trained first layer estimators.

        Parameters
        ----------
        X : DataFrame
            A pandas DataFrame containing the test data.

        Returns
        -------
        DataFrame:
            A pandas DataFrame containing the transformed test data.
        """

        # Determine which set of first layer estimators to use based on test_avg
        if self.test_avg:
            estimators_list = self.estimators_A_
        else:
            estimators_list = self.estimators_B_

        # Loop over each set of first layer estimators
        pred_list = []
        for estimators in estimators_list:
            # Check the average and method used by the first estimator
            avg, method = _check_avg(estimators[0], self.avg_type, self.method)

            # Make predictions on the input samples in parallel
            preds = Parallel(n_jobs=self.n_jobs)(
                (
                    delayed(_predict)(estimator, method, X, self._y)
                    for estimator in estimators
                )
            )

            # Average the predictions and append them to the list
            pred = _avg_preds(preds, avg, X, self._y)
            pred_list.append(pred)

        # Stack the predictions from each set of first layer estimators
        S = stack_preds(pred_list, self.names_)
        return S

    def set_params(self, **params: dict) -> "StackingTransformer":
        """
        Sets the hyperparameters of the stacked estimator.

        Parameters
        ----------
        **params:
            A dictionary containing the hyperparameters to be set.

        Returns
        -------
        StackingTransformer:
            The stacked estimator with the updated hyperparameters.
        """
        return self._set_params("estimators", **params)

    def get_params(self, deep: bool = True) -> dict:
        """
        Gets the hyperparameters of the stacked estimator.

        Parameters
        ----------
        deep : bool
            A boolean indicating whether to get the hyperparameters of the base estimators as well.

        Returns
        -------
        dict:
            A dictionary containing the hyperparameters of the stacked estimator.
        """
        return self._get_params("estimators", deep=deep)


class StackingRegressor(StackingTransformer, RegressorMixin):
    """
    A Stacking Regressor that performs stacking with inbuilt cross-validation.

    Parameters
    ----------
    estimators : list
        List of (name, estimator) tuples (implementing fit/predict).
    meta_estimator : estimator object
        A regressor that is used to make predictions based on the predictions of the first layer estimators.
    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy:

        - None: use non-CV predictions
        - Integer: to specify the number of folds in a `(Stratified)KFold`.
        - An iterable: yielding (train, test) splits as arrays of indices.
    scoring : string, callable or None, optional, default: None
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only a single value.
        If None, the estimator's default scorer (if available) is used.
    test_avg : bool (default=True)
        Stacking strategy. See 'crossval' from 'robusta.crossval' for details.
    join_X : bool (default=False)
        If True, concatenate stacked predictions with the original data.
    n_jobs : int or None, optional (default=-1)
        Number of jobs to run in parallel. None means 1.
    verbose : int (default=1)
        Verbosity level.
    n_digits : int (default=4)
        Verbose score precision.

    Attributes
    ----------
    estimators : list
        List of (name, estimator) tuples.
    meta_estimator : estimator object
        A regressor that is used to make predictions based on the predictions of the first layer estimators.
    cv : int, cross-validation generator or an iterable
        The cross-validation splitting strategy.
    scoring : string, callable or None
        A string or a scorer callable object / function with signature ``scorer(estimator, X, y)`` which should
        return only a single value.
    test_avg : bool
        Stacking strategy.
    join_X : bool
        If True, concatenate stacked predictions with the original data.
    n_jobs : int or None
        Number of jobs to run in parallel.
    verbose : int
        Verbosity level.
    n_digits : int
        Verbose score precision.
    method : str
        A string indicating the method used to aggregate predictions.
    avg_type : str
        A string indicating the type of aggregation used to average predictions.
    """

    def __init__(
        self,
        estimators: List[BaseEstimator],
        meta_estimator: MetaEstimatorMixin,
        cv: int = 5,
        scoring: Optional[Union[str, Callable]] = None,
        test_avg: bool = True,
        join_X: bool = False,
        n_jobs: int = -1,
        verbose: int = 0,
        n_digits: int = 4,
        random_state: int = 0,
    ):
        self.estimators = estimators
        self.meta_estimator = meta_estimator
        self.cv = cv
        self.scoring = scoring

        self.test_avg = test_avg
        self.join_X = join_X

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_digits = n_digits
        self.random_state = random_state

        self.method = "predict"
        self.avg_type = "mean"

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None
    ) -> "StackingRegressor":
        """
        Fit the stacking regressor using training data.

        Parameters
        ----------
        X: pd.DataFrame
            The training input samples.
        y: pd.Series
            The target values.
        groups: pd.Series, optional
            Group labels for the samples, used to ensure that the same group labels
            are not present in both the training and validation sets.

        Returns
        -------
        self: StackingRegressor
            Returns self.
        """

        # Check that all base estimators are regressors
        _check_estimator_types(self.estimators, "regressor")

        # Check that all base estimators have unique names
        _check_estimator_names(self.estimators)

        # Save training data to be used later for generating predictions
        self._save_train(X, y)

        # Fit the base estimators using training data
        self._fit_1st_layer(X, y, groups)

        # Fit the meta estimator using predictions from the base estimators
        self._fit_2nd_layer(X, y, groups)

        return self

    def _fit_2nd_layer(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the meta estimator using predictions from the base estimators.

        Parameters
        ----------
        X: pd.DataFrame
            The training input samples.
        y: pd.Series
            The target values.
        """

        # Transform input data using base estimators
        S = self.transform(X)

        # Fit the meta estimator using transformed data
        self.meta_estimator_ = clone(self.meta_estimator).fit(S, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions for new data using the trained meta estimator.

        Parameters
        ----------
        X: pd.DataFrame
            The input data to generate predictions for.

        Returns
        -------
        y: pd.Series
            The predicted target values for the input data.
        """

        # Generate predictions from the base estimators for X
        S = self.transform(X)

        # Use the predictions as input to the meta estimator to generate final predictions
        y = self.meta_estimator_.predict(S)

        return y


class StackingClassifier(StackingTransformer, ClassifierMixin):
    """
    Stacking Transformer with inbuilt Cross-Validation

    Parameters
    ----------
    estimators : list
        List of (name, estimator) tuples (implementing fit/predict).

    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy:

        - None, use non-CV predictions
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An iterable yielding (train, test) splits as arrays of indices.

    scoring : string, callable or None, optional, default: None
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only a single value.
        If None, the estimator's default scorer (if available) is used.

    test_avg : bool (default=True)
        Stacking strategy (essential parameter).

        See robusta.crossval for details.

    avg_type : string, {'soft', 'hard', 'auto', 'rank'} (default='auto')
        Averaging strategy for aggregating different CV folds predictions.

        See 'crossval' from 'robusta.crossval' for details.

    method : {'predict', 'predict_proba'}, optional (defaul='predict')
        Invokes the passed method name of the passed estimators.

    join_X : bool (default=False)
        If True, concatenate stacked predictions with the original data

    n_jobs : int or None, optional (default=-1)
        Number of jobs to run in parallel. None means 1.

    verbose : int (default=1)
        Verbosity level

    n_digits : int (default=4)
        Verbose score precision

    """

    def __init__(
        self,
        estimators: List[BaseEstimator],
        meta_estimator: MetaEstimatorMixin,
        cv: int = 5,
        scoring: Optional[Union[str, Callable]] = None,
        test_avg: bool = True,
        avg_type: str = "auto",
        method: str = "predict",
        join_X: bool = False,
        n_jobs: int = -1,
        verbose: int = 0,
        n_digits: int = 4,
        random_state: int = 0,
    ):
        self.estimators = estimators
        self.meta_estimator = meta_estimator
        self.cv = cv
        self.scoring = scoring

        self.test_avg = test_avg
        self.avg_type = avg_type
        self.method = method
        self.join_X = join_X

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_digits = n_digits
        self.random_state = random_state

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None
    ) -> "StackingClassifier":
        """
        Fit the stacking classifier using training data.

        Parameters
        ----------
        X: pd.DataFrame
            The training input samples.
        y: pd.Series
            The target values.
        groups: pd.Series, optional
            Group labels for the samples, used to ensure that the same group labels
            are not present in both the training and validation sets.

        Returns
        -------
        self: StackingClassifier
            Returns self.
        """

        # Check that all base estimators are classifiers
        _check_estimator_types(self.estimators, "classifier")

        # Check that all base estimators have unique names
        _check_estimator_names(self.estimators)

        # Save training data to be used later for generating predictions
        self._save_train(X, y)

        # Fit the base estimators using training data
        self._fit_1st_layer(X, y, groups)

        # Fit the meta estimator using predictions from the base estimators
        self._fit_2nd_layer(X, y, groups)

        return self

    def _fit_2nd_layer(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the meta estimator using predictions from the base estimators.

        Parameters
        ----------
        X: pd.DataFrame
            The training input samples.
        y: pd.Series
            The target values.
        """

        # Transform input data using base estimators
        S = self.transform(X)

        # Fit the meta estimator using transformed data
        self.meta_estimator_ = clone(self.meta_estimator).fit(S, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained regression classifier.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples to make predictions on.

        Returns
        -------
        y: np.ndarray
            The predicted target values.
        """

        # Transform input data using base estimators
        S = self.transform(X)

        # Use the trained meta estimator to make predictions on transformed data
        y = self.meta_estimator_.predict(S)

        return y

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions using the trained stacking classifier.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples to make predictions on.

        Returns
        -------
        y: np.ndarray
            The predicted probability values.
        """

        # Transform input data using base estimators
        S = self.transform(X)

        # Use the trained meta estimator to make probability predictions on transformed data
        y = self.meta_estimator_.predict_proba(S)

        return y

    @property
    def classes_(self) -> np.ndarray:
        """
        Return the classes used by the meta estimator for predictions.

        Returns
        -------
        classes_: np.ndarray
            An array of class labels.
        """
        return self.meta_estimator_.classes_


def stack_preds(
    pred_list: List[Union[pd.DataFrame, pd.Series]], names: List[str]
) -> pd.DataFrame:
    """
    Combine a list of predictions into a single dataframe with column names derived from the input names.

    Parameters
    ----------
    pred_list : List[Union[pd.DataFrame, pd.Series]]
        A list of prediction dataframes or series.
    names : List[str]
        A list of names corresponding to the prediction dataframes or series in pred_list.

    Returns
    -------
    pred : DataFrame
        A concatenated dataframe containing all of the prediction data with column names derived from the input names.
    """

    # Iterate over the list of prediction dataframes or series and add a column name prefix based on the input names.
    for name, pred in zip(names, pred_list):
        if hasattr(pred, "columns"):  # Check if pred is a dataframe
            # Create a MultiIndex for the columns with a tuple for each column consisting of the column name and the
            # input name.
            cols = [(col, name) for col in pred.columns]
            cols = pd.MultiIndex.from_tuples(cols)
            pred.columns = cols
        else:  # If pred is a series, set the name to the input name.
            pred.name = name

    # Concatenate the list of prediction dataframes or series horizontally.
    pred = pd.concat(pred_list, axis=1)
    return pred


def stack_results(
    results: Dict[str, Dict[str, pd.DataFrame]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine the out-of-fold and new predictions of multiple models into two concatenated dataframes.

    Parameters
    ----------
    results : (Dict[str, Dict[str, pd.DataFrame]])
        A dictionary of dictionaries containing the out-of-fold predictions ('oof_pred') and
        new predictions ('new_pred') for multiple models. The outer dictionary keys are model names.

    Returns
    -------
    DataFrames : tuple
        - S_train (pd.DataFrame): A dataframe containing the out-of-fold predictions for each model,
          with column names derived from the model names.
        - S_test (pd.DataFrame): A dataframe containing the new predictions for each model, with column names
          derived from the model names.
    """
    # Extract the out-of-fold and new predictions for each model from the input dictionary.
    oof_preds = [result["oof_pred"].copy() for result in results.values()]
    new_preds = [result["new_pred"].copy() for result in results.values()]

    # Get the names of the models from the input dictionary.
    names = [result["model_name"] for result in results.values()]

    # Stack the out-of-fold and new predictions for each model into concatenated dataframes.
    S_train = stack_preds(oof_preds, names)
    S_test = stack_preds(new_preds, names)

    return S_train, S_test


def _check_estimator_types(
    estimators: List[BaseEstimator], allow_types: Optional[List[str]] = None
) -> None:
    """
    Check that all estimators in a list have the same estimator type, and that the estimator type is allowed.

    Parameters
    ----------
    estimators : List[BaseEstimator]
        A list of scikit-learn estimators.
    allow_types : Optional[List[str]]
        A list of allowed estimator types. Defaults to ['classifier', 'regressor'].

    Raises
    ------
    ValueError:
        If the estimator types in the list are not all the same, or if the estimator type is not allowed.
    """
    # Get the estimator types for each estimator in the input list.
    estimator_types = np.array([e._estimator_type for _, e in estimators])

    # If no allowed estimator types were provided, use the default list.
    if allow_types is None:
        allow_types = ["classifier", "regressor"]
    allow_types = np.array(allow_types)

    # Check that all estimator types in the input list are the same.
    if not (estimator_types == estimator_types[0]).all():
        raise ValueError("Estimator types must be the same")

    # Check that the estimator type is allowed.
    if not (allow_types == estimator_types[0]).any():
        raise ValueError("Estimator types must be in: {}".format(allow_types))


def _check_estimator_names(estimators: List[BaseEstimator]) -> None:
    """
    Checks that the names of the given estimators are unique.

    Parameters
    ----------
    estimators: List[BaseEstimator]
        A list of scikit-learn estimator objects

    Raises
    ------
    ValueError:
        If the estimator names are not unique.

    """

    # Extract names of the estimators
    names = np.array([name for name, _ in estimators])

    # Get unique names
    unames = np.unique(names)

    # Check if the unique names have the same shape as the original names
    # If not, it means some names were duplicated, hence not unique
    if unames.shape != names.shape:
        raise ValueError("Estimator names must be unique")
