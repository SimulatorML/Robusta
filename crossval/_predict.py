import gc
from time import time
from typing import Callable, List, Optional, Iterable, Tuple, Union, Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import mode
from sklearn.base import is_classifier, is_regressor, BaseEstimator
from sklearn.utils.metaestimators import _safe_split

from ..importance import get_importance


def _fit_predict(estimator: BaseEstimator,
                 method: str,
                 scorer: Optional[Callable],
                 X: pd.DataFrame,
                 y: pd.Series,
                 X_new: Optional[pd.DataFrame] = None,
                 new_index: Optional[pd.Series] = None,
                 trn: Optional[np.array] = None,
                 oof: Optional[np.array] = None,
                 return_estimator: bool = False,
                 return_pred: bool = False,
                 fold: Optional[int] = None,
                 logger: Optional[object] = None,
                 train_score: bool = False,
                 y_transform: Optional[Callable] = None) -> dict:
    """
    Fits an estimator on a subset of training data and returns the results of the fit, including predictions and scores
    on out-of-fold and new data.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to fit.
    method : str
        The method used to predict values. Must be one of {'predict', 'predict_proba', 'predict_log_proba'}.
    scorer : callable, optional
        The scoring function to use. If not provided, no scores are computed.
    X : pd.DataFrame
        The input data to fit on.
    y : pd.Series
        The target variable to fit on.
    X_new : pd.DataFrame, optional
        New data to predict on.
    new_index : pd.Series, optional
        The index of the new data.
    trn : np.ndarray, optional
        An array of indices representing the training data to use. If not provided, all data is used.
    oof : np.ndarray, optional
        An array of indices representing the out-of-fold data to use. If not provided, no out-of-fold predictions or
        scores are computed.
    return_estimator : bool, default=False
        Whether to return the fitted estimator.
    return_pred : bool, default=False
        Whether to return predictions on out-of-fold and new data.
    fold : int, optional
        The fold number of a cross-validation. If provided, the results are logged to the provided logger object.
    logger : object, optional
        An object with a 'log' method that logs the provided fold and results dictionary.
    train_score : bool, default=False
        Whether to compute the training score using the provided scoring function.
    y_transform : callable, optional
        A function to transform the target variable before fitting the estimator.

    Returns
    -------
    dict
        A dictionary containing the results of the fit, including the fit time, validation score, training score (if
        computed), out-of-fold predictions (if requested), new data predictions (if requested), and the fitted estimator
        (if requested).
    """
    # create an empty dictionary to store the results
    result = {}

    # create an array of indices for new data if it exists, otherwise create an array of length 0
    new = np.arange(X_new.shape[0]) if X_new is not None else np.arange(0)

    # create an array of indices for training data if it is not provided, otherwise use the provided array
    trn = np.arange(X.shape[0]) if trn is None else trn

    # create an array of indices for out-of-fold data if it is not provided, otherwise use the provided array
    oof = np.arange(0) if oof is None else oof

    # split the training data and labels using the provided indices
    X_trn, y_trn = _safe_split(estimator=estimator,
                               X=X,
                               y=y,
                               indices=trn)

    # split the out-of-fold data and labels using the provided indices
    X_oof, y_oof = _safe_split(estimator=estimator,
                               X=X,
                               y=y,
                               indices=oof)

    # determine the index for the out-of-fold data (using the index of X_oof if it exists, otherwise using the index
    # of y_oof)
    oof_index = getattr(X_oof, 'index', y_oof.index)

    # determine the index for new data (using the index of X_new if it exists, otherwise using the provided new_index)
    new_index = getattr(X_new, 'index', new_index)

    # transform the training labels using the provided function (if it exists), otherwise use the original labels
    y_trn_ = y_transform(y_trn) if y_transform else y_trn

    # record the time and fit the estimator on the training data
    tic = time()
    estimator.fit(X_trn, y_trn_)
    result['fit_time'] = time() - tic

    # if requested, record the estimator object
    if return_estimator:
        result['estimator'] = estimator
    # if possible, compute and record feature importance's
    try:
        result['importance'] = get_importance(estimator)
    except (Exception,):
        pass

    # if requested and there is out-of-fold or new data, record the predictions on each
    if return_pred and (len(oof) or len(new)):
        tic = time()
        if len(oof):
            oof_pred = _predict(estimator=estimator,
                                method=method,
                                X=X_oof,
                                y=y,
                                index=oof_index)
            result['oof_pred'] = oof_pred
        if len(new):
            new_pred = _predict(estimator=estimator,
                                method=method,
                                X=X_new,
                                y=y,
                                index=new_index)
            result['new_pred'] = new_pred
        result['pred_time'] = time() - tic

    # if requested and there is out-of-fold data, compute and record the validation score using the provided scoring
    # function
    tic = time()
    if scorer and len(oof):
        result['val_score'] = scorer(estimator,
                                     X_oof,
                                     y_oof)

    # if requested and train_score is True, compute and record the training score using the provided scoring function
    if scorer and train_score:
        result['trn_score'] = scorer(estimator,
                                     X_trn,
                                     y_trn)

    result['score_time'] = time() - tic

    # if a logger object is provided, log the fold and the results
    if logger:
        logger.log(fold, result)

    # return the results dictionary
    return result


def _predict(estimator: BaseEstimator,
             method: str,
             X: pd.DataFrame,
             y: pd.Series,
             index: Optional[Iterable] = None) -> pd.DataFrame:
    """
    Make predictions using an estimator and input data X.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to use for prediction.
    method : str
        The method to use for prediction (e.g., 'predict', 'predict_proba', 'decision_function').
    X : pd.DataFrame
        The input data for prediction.
    y : pd.Series
        The target data for prediction.
    index : Optional[Iterable], default=None
        The index to use for the output DataFrame.

    Returns
    -------
    pd.DataFrame
        The predicted values as a DataFrame.

    Raises
    ------
    AttributeError
        If the estimator does not have the specified prediction method or if it is a classifier
        and does not have a 'classes_' attribute.
    TypeError
        If the estimator is of unknown type.
    """

    # Create a new DataFrame using the target data y
    Y = pd.DataFrame(y)

    # Get the name of the estimator class
    name = estimator.__class__.__name__

    # Get the method to be used for prediction
    action = getattr(estimator,
                     method,
                     None)

    # If the method exists, perform the prediction using the estimator and input data X
    if action:
        P = action(X)

    # If the method does not exist, raise an AttributeError
    else:
        raise AttributeError("<{}> has no method <{}>".format(name, method))

    # Get the index from X or use the given index argument
    index = getattr(X, 'index', index)

    # If the prediction result is a list of arrays, convert each array to a DataFrame using the given index
    if isinstance(P, list):
        P = [pd.DataFrame(p, index=index) for p in P]

    # If the prediction result is a single array, convert it to a DataFrame using the given index
    else:
        P = [pd.DataFrame(P, index=index)]

    # Concatenate the DataFrames along axis=1
    P = pd.concat(P, axis=1)

    # If the method is 'predict', set the column names to the name of the target variable y or Y
    if method is 'predict':
        P.columns = [y.name] if hasattr(y, 'name') else Y.columns

    # If the estimator is a classifier, set the column names according to the class labels
    elif is_classifier(estimator):
        # If the estimator is a MultiOutputClassifier or MultiTargetClassifier, set the column names as a
        # MultiIndex with level names '_TARGET' and '_CLASS'
        if name in ['MultiOutputClassifier', 'MultiTargetClassifier']:
            Classes = estimator.classes_
            tuples = []
            for target, classes in zip(Y.columns, Classes):
                for c in classes:
                    tuples.append((target, c))
            P.columns = pd.MultiIndex.from_tuples(tuples, names=('_TARGET', '_CLASS'))

        # If the estimator has a 'classes_' attribute, set the column names as the class labels
        elif hasattr(estimator, 'classes_'):
            classes = estimator.classes_
            P.columns = classes

        # If the estimator does not have a 'classes_' attribute, raise an AttributeError
        else:
            msg = "Classifier <{}> should has <classes_> attribute!".format(name)
            raise AttributeError(msg)

    # If the estimator is not a classifier, raise a TypeError
    else:
        est_type = getattr(estimator, "_estimator_type", None)
        raise TypeError('<{}> is an estimator of unknown type: \
                         <{}>'.format(name, est_type))

    # Return the prediction result as a DataFrame
    return P


def _check_avg(estimator: BaseEstimator,
               avg_type: str,
               method: str) -> Tuple[Callable, str]:
    """
    Check the compatibility between the estimator, the averaging type and the method passed.

    Parameters
    ----------
    estimator: BaseEstimator
        An instance of the estimator class to be checked.
    avg_type: str
        The type of averaging to be used. Can be one of {'mean', 'soft', 'hard', 'rank', 'auto', 'pass'}.
    method: str
        The prediction method to be used. Can be one of {'predict', 'predict_proba', 'decision_function'}.

    Returns
    -------
    Tuple containing the averaging function to be used and the updated method if necessary.
    """
    avg = None

    # Get the name of the estimator's class
    name = estimator.__class__.__name__

    # Available prediction methods
    methods = ['predict', 'predict_proba']
    if method not in methods:
        # Raise an error if the passed method is not available
        raise ValueError("<method> should be in {}".format(set(methods)) \
                         + "\n\t\tPassed '{}'".format(method))

    # Available averaging types
    avg_types = ['mean', 'soft', 'hard', 'rank', 'auto', 'pass']
    if avg_type not in avg_types:
        # Raise an error if the passed averaging type is not available
        raise ValueError("<avg_type> should be in {}".format(set(avg_types)) \
                         + "\n\t\tPassed '{}'".format(avg_type))

    # Check if the estimator is a classifier and has the predict_proba attribute
    if is_classifier(estimator) and hasattr(estimator, 'predict_proba'):
        if method is 'predict_proba':
            # Define the averaging function according to the passed averaging type
            if avg_type is 'pass':
                avg = _pass_pred
            elif avg_type is 'rank':
                avg = _rank_pred
            elif avg_type in ['auto', 'mean']:
                avg = _mean_pred
            else:
                # Raise an error if the passed averaging type is not compatible with the passed method
                good_vals = {'mean', 'auto', 'pass'}
                bad_vals = {'soft', 'hard'}
                msg = "Selected <method> value is {}.".format(method) \
                      + "\n\t\tAvailable <avg_type> options are: {}.".format(good_vals) \
                      + "\n\t\tBut current <avg_type> value set to '{}'.".format(avg_type) \
                      + "\n\t\t" \
                      + "\n\t\tNote: {} are voting strategies, so".format(bad_vals) \
                      + "\n\t\tthey are available only for method='predict'."
                raise ValueError(msg)
        elif method is 'predict':
            # Define the averaging function and update the method according to the passed averaging type
            if avg_type in ['soft', 'auto']:
                method = 'predict_proba'
                avg = _soft_vote
            elif avg_type is 'hard':
                avg = _hard_vote
            elif avg_type is 'pass':
                avg = _pass_pred
            else:
                # Raise an error if the passed averaging type is not available for the passed method
                raise ValueError("Passed unavailable avg_type '{}' for method <{}>"
                                 "".format(avg_type, method))
        elif method is 'decision_function':
            # Define the averaging function according to the passed averaging type
            if avg_type in ['mean', 'auto']:
                avg = _mean_pred
            elif avg_type is 'rank':
                avg = _rank_pred
            elif avg_type is 'pass':
                avg = _pass_pred
            else:
                # Raise an error if the passed averaging type is not available for the passed method
                raise ValueError("Passed unavailable avg_type '{}' for method <{}>"
                                 "".format(avg_type, method))

    # Check if the estimator is a classifier and does not have the predict_proba attribute
    elif is_classifier(estimator) and not hasattr(estimator, 'predict_proba'):
        if method is 'predict_proba':
            # Raise an error because predict_proba is not available for the estimator
            msg = "<{}> is not available for <{}>".format(method, name)
            raise AttributeError(msg)
        elif method in ['predict', 'decision_function']:
            if avg_type in ['hard', 'auto']:
                avg = _hard_vote
            elif avg_type is 'pass':
                avg = _pass_pred
            else:
                vals = {'auto', 'hard', 'pass'}
                msg = "<{}> is a {}. ".format(name, 'non-probabilistic classifier') \
                      + "\n\t\tAvailable <avg_type> options are: {}".format(vals) \
                      + "\n\t\tCurrent value set to '{}'".format(avg_type)
                raise ValueError(msg)
    elif is_regressor(estimator):
        # If the estimator is a regressor, check the averaging type passed and define the averaging function accordingly
        if avg_type is 'pass':
            avg = _pass_pred
            method = 'predict'
        elif avg_type in ['mean', 'auto']:
            avg = _mean_pred
            method = 'predict'
        else:
            vals = {'mean', 'auto', 'pass'}
            msg = "<{}> is a {}. ".format(name, 'regressor') \
                  + "\n\t\tAvailable <avg_type> options are: {}".format(vals) \
                  + "\n\t\tCurrent value set to '{}'".format(avg_type)
            raise ValueError(msg)
    return avg, method


def _avg_preds(preds: List[pd.DataFrame],
               avg: Callable,
               X: pd.DataFrame,
               y: pd.Series,
               index: Optional[pd.Index] = None) -> pd.Series:
    """
    Average predictions across multiple models and return the result.

    Parameters
    ----------
    preds : list[pd.DataFrame]
        A list of DataFrames of model predictions.
    avg : Callable
        A callable function to apply to the predictions for averaging.
    X : pd.DataFrame
        A DataFrame of input features.
    y : pd.Series
        A Series of target values.
    index : pd.Index, optional
        An optional index of rows to include in the output. Defaults to None.

    Returns
    -------
    pd.Series:
        A Series of averaged model predictions.
    """

    # Get the index from X or use the provided index
    index = getattr(X, 'index', index)

    # Concatenate the predictions into a single DataFrame
    pred = pd.concat(preds, axis=1)

    # Select the rows based on the given index
    pred = pred.loc[index]

    # Free memory by deleting the original predictions and collecting garbage
    del preds
    gc.collect()

    # Apply the given averaging function to the predictions
    pred = avg(pred)

    # If the resulting DataFrame has only one column, convert it to a Series
    if hasattr(pred, 'columns') and pred.shape[1] == 1:
        pred = pred.iloc[:, 0]

    # Remove any predictions for the zero class
    pred = _drop_zero_class(pred, y)

    return pred


def _drop_zero_class(pred: Union[pd.DataFrame, pd.Series],
                     y: pd.Series) -> Union[pd.DataFrame, pd.Series]:
    """
    Remove any predictions for the zero class from a DataFrame or Series of predictions.

    Parameters
    ----------
    pred : Union[pd.DataFrame, pd.Series]
        A DataFrame or Series of model predictions.
    y : pd.Series
        A Series of target values.

    Returns
    -------
    Union[pd.DataFrame, pd.Series]:
        The input DataFrame or Series with zero-class predictions removed.
    """

    # If the input is a Series, return it as-is
    if len(pred.shape) < 2:
        return pred

    # If the input has more than one column and any column has multiple occurrences, return it as-is
    if (pred.columns.value_counts() > 1).any():
        return pred

    # If the input has multi-level columns, extract the predictions for the one class in binary classification problems
    if hasattr(pred.columns, 'levels'):
        # Get the unique target values from the column index
        targets = pred.columns.get_level_values(0).unique()

        # Extract the predictions for each target value
        preds = [pred.loc[:, target] for target in targets]

        # Check if all targets are binary classifications
        is_binary = [list(p.columns) in [['0', '1'], [0, 1]] for p in preds]
        is_binary = np.array(is_binary).all()

        # If so, extract the predictions for the one class and concatenate them into a new DataFrame
        if is_binary:
            preds = [p.loc[:, 1] for p in preds]
            pred = pd.concat(preds, axis=1)
            pred.columns = targets
            pred.columns.name = None

    # If the input has two columns named '0' and '1', extract the predictions for the one class
    elif list(pred.columns) == ['0', '1'] \
            or list(pred.columns) == [0, 1]:
        pred = pred.iloc[:, 1]
        pred.name = y.name

    return pred


def _pass_pred(pred: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """
    Return the input DataFrame or Series of predictions as-is.

    Parameters
    ----------
    pred : Union[pd.DataFrame, pd.Series]
        A DataFrame or Series of model predictions.

    Returns
    -------
    Union[pd.DataFrame, pd.Series]:
        The input DataFrame or Series of predictions.
    """

    # Return the input as-is
    return pred


def _mean_pred(pred: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the mean prediction across all models.

    If the input is a DataFrame with multiple targets, this function is called recursively on each target.

    Parameters
    ----------
    pred : Union[pd.DataFrame, pd.Series]
        A DataFrame or Series of model predictions.

    Returns
    -------
    Union[pd.DataFrame, pd.Series]:
        The mean prediction across all models.
    """

    # If the input has multiple targets, call this function recursively on each target
    if hasattr(pred.columns, 'levels'):
        return _multioutput_pred(pred=pred,
                                 avg=_mean_pred)

    # Otherwise, calculate the mean prediction across all models
    else:
        return pred.groupby(pred.columns, axis=1).mean()


def _rank_pred(pred: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the rank-based prediction across all models.

    If the input is a DataFrame with multiple targets, this function is called recursively on each target.

    Parameters
    ----------
    pred : Union[pd.DataFrame, pd.Series]
        A DataFrame or Series of model predictions.

    Returns
    -------
    Union[pd.DataFrame, pd.Series]:
        The rank-based prediction across all models.
    """

    # If the input has multiple targets, call this function recursively on each target
    if hasattr(pred.columns, 'levels'):
        return _multioutput_pred(pred=pred,
                                 avg=_rank_pred)

    # Otherwise, calculate the rank-based prediction across all models
    else:
        return pred.rank(pct=True).groupby(pred.columns, axis=1).mean()


def _soft_vote(pred: Union[pd.DataFrame, pd.Series]) -> DataFrame | Any:
    """
    Calculate the soft voting prediction across all models.

    If the input is a DataFrame with multiple targets, this function is called recursively on each target.

    Parameters
    ----------
    pred : Union[pd.DataFrame, pd.Series]
        A DataFrame or Series of model predictions.

    Returns
    -------
    Union[pd.Series]:
        The soft voting prediction across all models.
    """

    # If the input has multiple targets, call this function recursively on each target
    if hasattr(pred.columns, 'levels'):
        return _multioutput_vote(pred=pred,
                                 vote=_soft_vote)

    # Otherwise, calculate the soft voting prediction across all models
    else:
        return pred.groupby(pred.columns, axis=1).mean().idxmax(axis=1)


def _hard_vote(pred: Union[pd.DataFrame, pd.Series]) -> DataFrame | Any:
    """
    Calculate the hard voting prediction across all models.

    If the input is a DataFrame with multiple targets, this function is called recursively on each target.

    Parameters
    ----------
    pred : Union[pd.DataFrame, pd.Series]
        A DataFrame or Series of model predictions.

    Returns
    -------
    Union[pd.Series]:
        The hard voting prediction across all models.
    """

    # If the input has multiple targets, call this function recursively on each target
    if hasattr(pred.columns, 'levels'):
        return _multioutput_vote(pred=pred,
                                 vote=_hard_vote)

    # Otherwise, calculate the hard voting prediction across all models
    else:
        return pred.apply(lambda x: mode(x)[0][0], axis=1)


def _multioutput_vote(pred: pd.DataFrame,
                      vote: Callable) -> pd.DataFrame:
    """
    Apply the given voting function to each target in a multi-output prediction.

    This function is called by _soft_vote and _hard_vote when the input has multiple targets.

    Parameters
    ----------
    pred : pd.DataFrame
        A DataFrame of multi-output model predictions.
    vote : Callable
        The voting function to apply to each target.

    Returns
    -------
    pd.DataFrame:
        The prediction for each target after applying the voting function.
    """

    # Get the unique targets in the input
    targets = pred.columns.get_level_values(0).unique()

    # Apply the voting function to each target separately
    preds = [pred.loc[:, target] for target in targets]
    preds = [vote(p) for p in preds]

    # Combine the predictions for each target into a single DataFrame
    pred = pd.concat(preds, axis=1)
    pred.columns = targets
    pred.columns.name = None

    return pred


def _multioutput_pred(pred: pd.DataFrame,
                      avg: Callable) -> pd.DataFrame:
    """
    Apply the given averaging function to each target in a multi-output prediction.

    This function is called by _mean_pred and _rank_pred when the input has multiple targets.

    Parameters
    ----------
    pred : pd.DataFrame
        A DataFrame of multi-output model predictions.
    avg : Callable
        The averaging function to apply to each target.

    Returns
    -------
    pd.DataFrame:
        The prediction for each target after applying the averaging function.
    """

    # Get the unique columns and targets in the input
    cols = pred.columns.unique()
    targets = pred.columns.get_level_values(0).unique()

    # Apply the averaging function to each target separately
    preds = [pred.loc[:, target] for target in targets]
    preds = [avg(p) for p in preds]

    # Combine the predictions for each target into a single DataFrame
    pred = pd.concat(preds, axis=1)
    pred.columns = cols

    return pred
