from datetime import datetime
from time import time
from typing import Callable, Iterable, Optional, Any, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import clone
from sklearn.base import is_classifier, BaseEstimator
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.utils import indexable

from . import CVLogger
from . import _check_avg, _fit_predict, _avg_preds
from ..utils import ld2dl, logmsg


def copy(estimator: BaseEstimator) -> object | Any:
    """
    A function that returns a copy of an estimator.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator object to be copied.

    Returns
    -------
    object:
        A copy of the estimator object.
    """

    # Check if the estimator object has a 'copy' method
    if hasattr(estimator, 'copy'):
        # If it does, use the 'copy' method to create a copy
        return estimator.copy()
    else:
        # If it doesn't, use the 'clone'  to create a copy
        return clone(estimator)


def crossval(estimator: BaseEstimator,
             cv: int,
             X: pd.DataFrame,
             y: pd.Series,
             groups: Optional[pd.Series] = None,
             X_new: Optional[pd.DataFrame] = None,
             new_index: Optional[Iterable] = None,
             scoring: Optional[Union[str, Callable]] = None,
             test_avg: bool = True,
             avg_type: str = 'auto',
             method: str = 'predict',
             return_pred: bool = True,
             return_estimator: bool = False,
             verbose: int = 2,
             n_digits: int = 4,
             n_jobs: Optional[int] = None,
             compact: bool = False,
             train_score: bool = False,
             y_transform: Optional[Callable] = None,
             **kwargs) -> dict:
    """
    Perform cross-validation for an estimator and return the results.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to fit.
    cv : int
        Number of folds in cross-validation.
    X : pd.DataFrame
        The feature data.
    y : pd.Series
        The target data.
    groups : Optional[pd.Series], default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Can be used to ensure that the same group is not in
        both testing and training set. When `None`, the training and testing
        splits are made by preserving the percentage of samples for each class.
    X_new : Optional[pd.DataFrame], default=None
        New data to predict on after fitting the estimator.
    new_index : Optional[Iterable], default=None
        The index of the new data.
    scoring : Optional[str, Callable], default=None
        Scoring metric to use for cross-validation.
    test_avg : bool, default=True
        Whether to average the test predictions across folds.
    avg_type : str, default='auto'
        The type of averaging to use for test predictions. If 'auto', it
        will choose the appropriate averaging method based on the estimator
        and scoring metric. Valid options are 'mean', 'median', 'harmonic',
        and 'geometric'.
    method : str, default='predict'
        The method used to generate predictions. If 'predict', use
        estimator.predict(); if 'predict_proba', use estimator.predict_proba().
    return_pred : bool, default=True
        Whether to return the predicted values.
    return_estimator : bool, default=False
        Whether to return the estimator object fitted on the full training set.
    verbose : int, default=2
        Controls the verbosity when fitting and predicting.
    n_digits : int, default=4
        The number of digits to use when logging floats.
    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel for cross-validation. If `None`,
        it will use all available CPUs.
    compact : bool, default=False
        Whether to log results in a compact format.
    train_score : bool, default=False
        Whether to compute the train score as well.
    y_transform : Optional[Callable], default=None
        A function to apply to the target data before fitting.

    Returns
    -------
    result : dict
        A dictionary containing the results of the cross-validation. The keys
        are 'oof_pred', 'new_pred', 'importance', 'val_score', 'trn_score',
        'fit_time', 'score_time', 'pred_time', 'concat_time', 'features',
        'datetime', 'scorer', and 'cv'.
    """
    # make sure X, y, and groups are indexable
    X, y, groups = indexable(X, y, groups)

    # make sure X_new is indexable
    X_new, _ = indexable(X_new, None)

    # check cross-validation strategy
    cv = check_cv(cv=cv,
                  y=y,
                  classifier=is_classifier(estimator))

    # check averaging type and prediction method
    avg, method = _check_avg(estimator=estimator,
                             avg_type=avg_type,
                             method=method)

    # check scoring metric
    scorer = check_scoring(estimator=estimator,
                           scoring=scoring)

    # create logger for cross-validation
    logger = CVLogger(estimator=estimator,
                      cv=cv,
                      verbose=verbose,
                      n_digits=n_digits,
                      compact=compact)

    # start logging
    logger.start()

    # create a parallel instance
    parallel = Parallel(max_nbytes='256M',
                        pre_dispatch='2*n_jobs',
                        n_jobs=n_jobs,
                        require='sharedmem')

    if test_avg:
        # run parallel computation
        result = parallel(
            # run _fit_predict function with specified parameters
            delayed(_fit_predict)(
                copy(estimator), method, scorer, X, y, X_new, new_index,
                trn, oof, return_estimator, return_pred, fold, logger,
                train_score, y_transform)
            # loop through cross-validation folds
            for fold, (trn, oof) in enumerate(cv.split(X, y, groups)))

        # convert list of dictionaries to dictionary of lists
        result = ld2dl(result)

    else:
        # run parallel computation
        result = parallel(
            # run _fit_predict function with specified parameters
            (delayed(_fit_predict)(
                copy(estimator), method, scorer, X, y, None, None, trn, oof,
                return_estimator, return_pred, fold, logger, train_score,
                y_transform)
                # loop through cross-validation folds
                for fold, (trn, oof) in enumerate(cv.split(X, y, groups))))

        if verbose >= 2:
            print()
            logmsg('Fitting full train set...')

        # run _fit_predict function with specified parameters
        result_new = _fit_predict(estimator=copy(estimator),
                                  method=method,
                                  scorer=None,
                                  X=X,
                                  y=y,
                                  X_new=X_new,
                                  new_index=new_index,
                                  trn=None,
                                  oof=None,
                                  return_estimator=return_estimator,
                                  return_pred=return_pred,
                                  fold=-1,
                                  logger=logger,
                                  train_score=train_score,
                                  y_transform=y_transform)

        # convert list of dictionaries to dictionary of lists
        result = ld2dl(ld=result)

        # loop through dictionary of results
        for key, val in result_new.items():

            # if the key exists in the result dictionary
            if key in result:
                # append the value to the existing list
                result[key].append(val)
            # if the key does not exist in the result dictionary
            else:
                # add the key and value to the result dictionary
                result[key] = [val]

    # create list of result keys that need concatenation
    needs_concat = ['oof_pred', 'new_pred', 'importance', 'val_score', 'trn_score']

    # if any of the keys in needs_concat are in the result dictionary
    if np.any(np.in1d(needs_concat, list(result))):

        # start timer
        tic = time()

        # if 'oof_pred' is a key in the result dictionary
        if 'oof_pred' in result:
            # get the list of oof predictions
            oof_preds = result['oof_pred']

            oof_pred = _avg_preds(preds=oof_preds,
                                  avg=avg,
                                  X=X,
                                  y=y,
                                  index=y.index)

            result['oof_pred'] = oof_pred
        if 'new_pred' in result:
            # get the list of new predictions
            new_preds = result['new_pred']

            new_pred = _avg_preds(preds=new_preds,
                                  avg=avg,
                                  X=X_new,
                                  y=y,
                                  index=new_index)

            result['new_pred'] = new_pred

        for key in ['fit_time', 'score_time', 'pred_time']:
            if key in result:
                result[key] = np.array(result[key])

        result['concat_time'] = time() - tic

    # add additional information to the result dictionary
    if hasattr(X, 'columns'): result['features'] = list(X.columns.values)

    result['datetime'] = datetime.now()
    result['scorer'] = scorer
    result['cv'] = cv

    # log the results and update the dictionary with any additional keyword arguments
    logger.end(result)
    result.update(kwargs)

    # return the result dictionary
    return result


def crossval_score(estimator: BaseEstimator,
                   cv: int,
                   X: pd.DataFrame,
                   y: pd.Series,
                   groups: Optional[pd.Series] = None,
                   scoring: Optional[Union[str, Callable]] = None,
                   n_jobs: Optional[int] = None,
                   verbose: int = 2,
                   n_digits: int = 4,
                   compact: bool = False,
                   train_score: bool = False,
                   target_func: Optional[Callable] = None) -> np.ndarray:
    """
    Perform cross-validation on the given estimator and return the scores.

    Parameters
    ----------
    estimator: BaseEstimator
        A scikit-learn estimator object implementing 'fit' and 'predict' methods.

    cv: int
        The number of folds to be used for cross-validation.

    X: pd.DataFrame
        The feature matrix.

    y: pd.Series
        The target variable.

    groups: Optional[pd.Series], default=None
        An array of group labels used for group-based cross-validation.

    scoring: Optional[str, Callable], default=None
        The scoring metric to be used. If None, estimator's default scoring method is used.

    n_jobs: Optional[int], default=None
        The number of CPUs to use for parallel computation. -1 means use all processors.

    verbose: int, default=2
        The verbosity level.

    n_digits: int, default=4
        The number of decimal places to be used in printing the scores.

    compact: bool, default=False
        Whether to print the scores in compact format or not.

    train_score: bool, default=False
        Whether to return training scores or not.

    target_func: Optional[Callable], default=None
        A callable function that can be used to transform the target variable.

    Returns
    -------
    np.ndarray
        The array of cross-validation scores.
    """

    # call the crossval function with the given parameters and return_pred argument set to False
    result = crossval(estimator=estimator,
                      cv=cv,
                      X=X,
                      y=y,
                      groups=groups,
                      n_digits=n_digits,
                      scoring=scoring,
                      n_jobs=n_jobs,
                      verbose=verbose,
                      return_pred=False,
                      compact=compact,
                      train_score=train_score,
                      target_func=target_func)

    # extract the validation scores from the result dictionary and return them as a numpy array
    scores = result['val_score']
    return scores


def crossval_predict(estimator: BaseEstimator,
                     cv: int,
                     X: pd.DataFrame,
                     y: pd.Series,
                     groups: Optional[pd.Series] = None,
                     X_new: Optional[pd.DataFrame] = None,
                     test_avg: bool = True,
                     avg_type: str = 'auto',
                     method: str = 'predict',
                     scoring: Optional[Union[str, Callable]] = None,
                     n_jobs: Optional[int] = None,
                     verbose: int = 0,
                     n_digits: int = 4,
                     compact: bool = False,
                     train_score: bool = False,
                     target_func: Optional[Callable] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Perform cross-validation and return out-of-fold predictions and predictions for new data.

    Parameters
    -----------
    estimator: BaseEstimator
        The estimator to be evaluated.
    cv: int
        Number of folds in cross-validation.
    X: pd.DataFrame
        The input data.
    y: pd.Series
        The target values.
    groups: pd.Series, optional (default=None)
        The group labels for the samples, used for grouping the samples when splitting the dataset into folds.
    X_new: pd.DataFrame, optional (default=None)
        The new data for which to generate predictions.
    test_avg: bool, optional (default=True)
        Whether to average test predictions across folds.
    avg_type: str, optional (default='auto')
        The type of averaging to perform. Possible values are 'macro', 'micro', 'samples', 'weighted', and 'auto'.
        The default value 'auto' chooses 'macro' for multiclass classification and regression, and 'binary' for
        binary classification.
    method: str, optional (default='predict')
        The name of the method to call on the estimator for generating predictions. This should be a string
        corresponding to a method of the estimator that takes X as input and returns an array of predicted values.
    scoring: str or callable, optional (default=None)
        The scoring function to use for evaluating the estimator. If None, the estimator's default scoring method
        is used.
    n_jobs: int, optional (default=None)
        The number of CPU cores to use for parallelization. If None, all available cores are used.
    verbose: int, optional (default=0)
        Verbosity level.
    n_digits: int, optional (default=4)
        The number of digits to include in floating point values.
    compact: bool, optional (default=False)
        Whether to return a compact representation of the results.
    train_score: bool, optional (default=False)
        Whether to compute the train score in addition to the test score.
    target_func: callable, optional (default=None)
        A function to apply to the target values before passing them to the estimator.

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        A tuple containing the out-of-fold predictions and predictions for new data, respectively.
    """

    # Call crossval function to perform cross-validation
    result = crossval(estimator=estimator,
                      cv=cv,
                      X=X,
                      y=y,
                      groups=groups,
                      X_new=X_new,
                      scoring=scoring,
                      avg_type=avg_type,
                      method=method,
                      test_avg=test_avg,
                      n_jobs=n_jobs,
                      verbose=verbose,
                      n_digits=n_digits,
                      compact=compact,
                      train_score=train_score,
                      target_func=target_func)

    # Extract out-of-fold and new predictions from cross-validation results
    oof_pred = result['oof_pred'] if 'oof_pred' in result else None
    new_pred = result['new_pred'] if 'new_pred' in result else None

    # Return predictions as a tuple
    return oof_pred, new_pred
