import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from datetime import datetime
from time import time

from sklearn.base import clone, is_classifier
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring
from robusta.utils import logmsg, ld2dl
from sklearn.utils import indexable

from ._predict import _fit_predict, _check_avg, _avg_preds
from ._output import CVLogger


__all__ = [
    'crossval',
    'crossval_score',
    'crossval_predict',
]




def crossval(estimator, cv, X, y, groups=None, X_new=None, scoring=None,
             test_avg=True, avg_type='auto', method='predict', return_pred=True,
             return_estimator=False, verbose=2, n_digits=4, n_jobs=-1,
             random_state=0):
    """Evaluate metric(s) by cross-validation and also record fit/score time,
    feature importances and compute out-of-fold and test predictions.

    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data.

    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

    X : DataFrame, shape [n_samples, n_features]
        The data to fit, score and calculate out-of-fold predictions

    y : Series, shape [n_samples]
        The target variable to try to predict

    groups : None
        Group labels for the samples used while splitting the dataset into
        train/test set

    X_new : DataFrame, shape [m_samples, n_features] or None
        The unseed data to predict (test set)

    test_avg : bool
        Stacking strategy (essential parameter)

        - True: bagged predictions for test set (given that we have N folds,
                we fit N models on each fold's train data, then each model
                predicts test set, then we perform bagging: compute mean of
                predicted values (for regression or class probabilities) - or
                majority vote: compute mode (when predictions are class labels)

        - False: predictions for tests set (estimator is fitted once on full
                 train set, then predicts test set)

        Ignored if return_pred=False or X_new is not defined.

    scoring : string, callable or None, optional, default: None
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only a single value.
        If None, the estimator's default scorer (if available) is used.

    avg_type : string, {'soft', 'hard', 'auto', 'rank', 'pass'} (default='auto')
        Averaging strategy for aggregating different CV folds predictions

        - 'hard' : use predicted class labels for majority rule voting.

                   Ignored if estimator type is 'regressor'.
                   Ignored if <return_pred> set to False.
                   Ignored if <method> is not 'predict'.

        - 'soft' : predicts the class label based on the argmax of the sums
                   of the predicted probabilities, which is recommended for
                   an ensemble of well-calibrated classifiers.

                   Ignored if estimator type is 'regressor'.
                   Ignored if <return_pred> set to False.
                   Ignored if <method> is not 'predict'.

        - 'auto' : use simple averaging for regressor's predcitions and for
                   classifier's probabilities (if <method> is 'predict_proba');

                   if estimator type is 'classifier' and <method> is 'predict',
                   set <averaging> to 'soft' for classifier with <predict_proba>
                   attribute, set <averaging> to 'hard' for other.

                   Ignored if <return_pred> set to False.

        - 'rank' : ranking probabilities along fold and averaging.

                   Prefered for scoring like 'AUC-ROC'.

        - 'pass' : leave predictions of different folds separated.

                   Column '_FOLD' will be added.

        Ignored if <return_pred> set to False, or <method> is not 'predict'.

    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.

        Ignored if return_pred=False.

    return_pred : bool (default=False)
        Return out-of-fold predictions (and test predictions, if X_new is defined)

    return_estimator : bool (default=False)
        Return fitted estimators

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int (default=1)
        Verbosity level

    n_digits : int (default=4)
        Verbose score(s) precision

    random_state : int (default=0)
        Cross-Validation seed


    Returns
    -------
    result : dict of array, float or Series
        Array of scores/predictions/time of the estimator for each run of the
        cross validation. If test_avg=True, arrays has shape [n_splits],
        otherwise [n_splits+1] except score & score_time.

        The possible keys for this ``dict`` are:

            ``fold`` : list of pair of list
                Two lists with trn/oof indices

            ``score`` : array or dict of array, shape [n_splits]
                The score array for test scores on each cv split.
                If multimetric, return dict of array.

            ``oof_pred`` : Series, shape [n_samples]
                Out-of-fold predictions.
                Ignored if return_pred=False.

            ``new_pred`` : Series, shape [m_samples]
                Test predictions (unseen data).
                Ignored if return_pred=False.

            ``fit_time`` : array of float, shape [n_splits] or [n_splits+1]
                The time for fitting the estimator on the train
                set for each cv split.

            ``pred_time`` : array of float, shape [n_splits] or [n_splits+1]
                Out-of-fold and test predictions time.
                Ignored if return_pred=False.

            ``score_time`` : array of float, shape [n_splits]
                Out-of-fold scores time for each cv split.

            ``concat_time`` : float
                Extra time spent on concatenation of predictions, importances
                or scores dictionaries. Ignored if all of return_pred,
                return_importance, return_score are set to False.

            ``estimator`` : list of estimator object, shape [n_splits] or [n_splits+1]
                The fitted estimator objects for each cv split (and ).
                Ignored if return_estimator=False.

            ``importance`` : Series, shape [n_features]
                Averaged <feature_importances_> or <coef_> of all estimators.
                Ignored if return_importance=False.

            ``features`` : list, shape [n_features]
                List of features.

    """

    # Check parameters
    X, y, groups = indexable(X, y, groups)
    X_new, _ = indexable(X_new, None)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    if hasattr(cv, 'random_state'):
        cv.random_state = random_state

    avg, method = _check_avg(estimator, avg_type, method)
    scorer = check_scoring(estimator, scoring)

    # Fit & predict
    logger = CVLogger(cv, verbose, prec=n_digits)
    logger.log_start(estimator, scorer)

    parallel = Parallel(max_nbytes='256M', pre_dispatch='2*n_jobs',
                        n_jobs=n_jobs, require='sharedmem')

    if test_avg:

        # Stacking Type A (test averaging = True)
        result = parallel(
            delayed(_fit_predict)(
                clone(estimator), method, scorer, X, y, X_new, trn, oof,
                return_pred, return_estimator, i, logger)
            for i, (trn, oof) in enumerate(cv.split(X, y, groups)))

        result = ld2dl(result)

    else:

        # Stacking Type B (test_averaging = False)
        result = parallel(
            (delayed(_fit_predict)(
                clone(estimator), method, scorer, X, y, None, trn, oof,
                return_pred, return_estimator, i, logger)
            for i, (trn, oof) in enumerate(cv.split(X, y, groups))))

        if verbose >= 2:
            print()
            logmsg('Fitting full train set...')

        result_new = _fit_predict(clone(estimator), method, None, X, y, X_new,
            None, None, return_pred, return_estimator, -1, None)

        result = ld2dl(result)
        for key, val in result_new.items():
            if key in result:
                result[key].append(val)
            else:
                result[key] = [val]


    # Concat Predictions (& Feature Importances)
    needs_concat = ['oof_pred', 'new_pred', 'importance', 'score']
    if np.any(np.in1d(needs_concat, list(result))):

        tic = time()

        if 'oof_pred' in result:
            oof_preds = result['oof_pred']
            oof_pred = _avg_preds(oof_preds, avg, X, y)
            result['oof_pred'] = oof_pred

        if 'new_pred' in result:
            new_preds = result['new_pred']
            new_pred = _avg_preds(new_preds, avg, X, y)
            result['new_pred'] = new_pred

        if 'importance' in result:
            importance = np.array(result['importance']).T
            importance = pd.DataFrame(importance, index=X.columns)
            result['importance'] = importance

        for key in ['fit_time', 'score_time', 'pred_time']:
            if key in result:
                result[key] = np.array(result[key])

        result['concat_time'] = time() - tic

    result['datetime'] = datetime.now()
    result['features'] = list(X.columns)
    result['cv'] = cv

    # Final score
    logger.log_final(result)

    return result




def crossval_score(estimator, cv, X, y, groups=None, scoring=None, n_jobs=-1,
                   verbose=2, n_digits=4, random_state=0):
    """Evaluate metric(s) by cross-validation and also record fit/score time,
    feature importances and compute out-of-fold and test predictions.

    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data.

    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

    X : DataFrame, shape [n_samples, n_features]
        The data to fit, score and calculate out-of-fold predictions

    y : Series, shape [n_samples]
        The target variable to try to predict

    groups : None
        Group labels for the samples used while splitting the dataset into
        train/test set

    scoring : string, callable or None, optional, default: None
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only a single value.
        If None, the estimator's default scorer (if available) is used.
        Ignored if return_score=False.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int (default=1)
        Verbosity level

    n_digits : int (default=4)
        Verbose score(s) precision

    random_state : int (default=0)
        Cross-Validation seed


    Returns
    -------
    scores : list of float
        Rows are splits. If multimetric, return DataFrame, where each column
        represents different metric.

    """
    result = crossval(estimator, cv, X, y, groups, n_digits=n_digits,
                      scoring=scoring, n_jobs=n_jobs, verbose=verbose,
                      random_state=random_state)

    scores = result['score']
    return scores




def crossval_predict(estimator, cv, X, y, groups=None, X_new=None, test_avg=True,
                     avg_type='auto', method='predict', scoring=None, n_jobs=-1,
                     verbose=0, n_digits=4, random_state=0):
    """Get Out-of-Fold and Test predictions.

    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data.

    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    X : DataFrame, shape [n_samples, n_features]
        The data to fit, score and calculate out-of-fold predictions

    y : Series, shape [n_samples]
        The target variable to try to predict

    groups : None
        Group labels for the samples used while splitting the dataset into
        train/test set

    X_new : DataFrame, shape [m_samples, n_features] or None
        The unseed data to predict (test set)

    test_avg : bool
        Stacking strategy (essential parameter)

        - True: bagged predictions for test set (given that we have N folds,
                we fit N models on each fold's train data, then each model
                predicts test set, then we perform bagging: compute mean of
                predicted values (for regression or class probabilities) - or
                majority vote: compute mode (when predictions are class labels)

        - False: predictions for tests set (estimator is fitted once on full
                 train set, then predicts test set)

    avg_type : string, {'soft', 'hard', 'rank', 'auto', 'pass'} (default='auto')
        Averaging strategy for aggregating different CV folds predictions

        - 'hard' : use predicted class labels for majority rule voting.

                   Ignored if estimator type is 'regressor'.
                   Ignored if <method> is not 'predict'.

        - 'soft' : predicts the class label based on the argmax of the sums
                   of the predicted probabilities, which is recommended for
                   an ensemble of well-calibrated classifiers.

                   Ignored if estimator type is 'regressor'.
                   Ignored if <method> is not 'predict'.

        - 'rank' : ranking probabilities along fold and averaging.

                   Prefered for scoring like 'AUC-ROC'.

        - 'auto' : use simple averaging for regressor's predcitions and for
                   classifier's probabilities (if <method> is 'predict_proba');

                   if estimator type is 'classifier' and <method> is 'predict',
                   set <averaging> to 'soft' for classifier with <predict_proba>
                   attribute, set <averaging> to 'hard' for other.

        - 'pass' : leave predictions of different folds separated.

                   Column '_FOLD' will be added.

        Ignored if <return_pred> set to False, or <method> is not 'predict'.

    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.

    scoring : string, callable or None, optional, default: None
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only a single value.
        If None, the estimator's default scorer (if available) is used.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int (default=1)
        Verbosity level

    n_digits : int (default=4)
        Verbose score(s) precision

    random_state : int (default=0)
        Cross-Validation seed


    Returns
    -------
    oof_pred : Series, shape [n_samples]
        Out-of-fold predictions

    new_pred : Series, shape [m_samples] or None
        Test predictions (unseen data)
        None if X_new is not defined

    """
    result = crossval(estimator, cv, X, y, groups, X_new=X_new, scoring=scoring,
                      avg_type=avg_type, method=method, test_avg=test_avg,
                      n_jobs=n_jobs, verbose=verbose, n_digits=n_digits,
                      random_state=random_state)

    oof_pred = result['oof_pred'] if 'oof_pred' in result else None
    new_pred = result['new_pred'] if 'new_pred' in result else None

    return oof_pred, new_pred
