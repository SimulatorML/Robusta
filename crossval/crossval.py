import pandas as pd
import numpy as np

from joblib import Parallel, delayed
import time

from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.model_selection import check_cv
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils import indexable

from robusta.preprocessing import LabelEncoder1D, LabelEncoder
from robusta.importance import extract_importance
from robusta.model import extract_model_name
from robusta.utils import logmsg, ld2dl

from ._output import CVLogger


__all__ = ['crossval', 'crossval_score', 'crossval_predict']




def crossval(estimator, cv, X, y, groups=None, X_new=None, test_avg=True,
             scoring=None, averaging='auto', method='predict', return_pred=True,
             return_estimator=True, return_score=True, return_importance=False,
             return_encoder=False, return_folds=True, n_jobs=-1, verbose=1,
             n_digits=4):
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
        Ignored if return_score=False.

    averaging : string, {'soft', 'hard', 'auto', 'pass'} (default='auto')
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

        - 'pass' : leave predictions of different folds separated.

                   Column 'fold' will be added.

        Ignored if <return_pred> set to False, or <method> is not 'predict'.

    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.
        Ignored if return_pred=False.

    return_pred : bool (default=False)
        Return out-of-fold predictions (and test predictions, if X_new is defined)

    return_estimtor : bool (default=False)
        Return fitted estimators

    return_score : bool (default=True)
        Return out-of-fold scores

    return_importance : bool (default=False)
        Return averaged <feature_importances_> or <coef_> of all estimators

    return_encoder : bool (default=False)
        Return label encoder for target

    return_folds : bool (default=False)
        Return folds

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int (default=1)
        Verbosity level

    n_digits : int (default=4)
        Verbose score(s) precision


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
                Ignored if return_score=False AND verbose=0.

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
                Ignored if return_score=False.

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

            ``encoder`` : transformer object or None
                The fitted target transformer. For classification task only,
                otherwise is None.

    """

    # Check data
    X, y, groups = indexable(X, y, groups)
    X_new, _ = indexable(X_new, None)


    # Check validation scheme
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    if hasattr(cv, 'random_state') and cv.random_state is None:
        cv.random_state = 0

    folds = list(cv.split(X, y, groups))


    # Check scorer(s)
    scorers, _ = _check_multimetric_scoring(estimator, scoring)

    if isinstance(scoring, str): # single metric case
        scorers = {scoring: scorers['score']}

    return_score = True if verbose else return_score


    # Check averaging strategy & method
    if return_pred:
        method, avg = _check_avg(estimator, averaging, method)
    else:
        method, avg = None, None


    # Init Estimator
    '''params = estimator.get_params()
    params_update = {}

    for key, val in params.items():
        # Parallel CV vs Parallel Model
        if key.endswith('n_jobs'):
            params_update[key] = 1 if n_jobs not in [None, 1] else val
        # Fix random seed
        if key.endswith('random_state'):
            params_update[key] = 0 if val is None else val
        # Verbosity level
        if key.endswith('verbose'):
            params_update[key] = 0 if verbose < 10 else val

    estimator = estimator.set_params(**params_update)'''

    # Init Logger
    logger = CVLogger(folds, verbose, prec=n_digits)
    logger.log_start(estimator, scorers)


    # Target Encoding
    if is_classifier(estimator):
        if len(y.shape) == 1:
            encoder = LabelEncoder1D()
            y = encoder.fit_transform(y)
        else:
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)

    else:
        encoder = None


    # Fit & predict
    parallel = Parallel(max_nbytes='256M', pre_dispatch='2*n_jobs',
                        n_jobs=n_jobs, require='sharedmem')

    if test_avg:

        # Stacking Type A (test averaging = True)
        result = parallel(
            delayed(_fit_pred_score)(clone(estimator), method, scorers, X, y,
                trn, oof, X_new, return_pred, return_estimator, return_score,
                return_importance, i, logger)
            for i, (trn, oof) in enumerate(folds))

        result = ld2dl(result)

    else:

        # Stacking Type B (test_averaging = False)
        result = parallel(
            (delayed(_fit_pred_score)(clone(estimator), method, scorers, X, y,
                trn, oof, None, return_pred, return_estimator, return_score,
                return_importance, i, logger)
            for i, (trn, oof) in enumerate(folds)))

        if verbose >= 2:
            print()
            logmsg('Fitting full train set...')

        result_new = _fit_pred_score(clone(estimator), method, None, X, y,
            None, None, X_new, return_pred, return_estimator, False,
            return_importance, -1, None)

        result = ld2dl(result)
        for key, val in result_new.items():
            if key in result:
                result[key].append(val)
            else:
                result[key] = [val]


    # Concat Predictions (& Feature Importances)
    needs_concat = ['oof_pred', 'new_pred', 'importance', 'score']
    if np.any(np.in1d(needs_concat, list(result))):

        start_time = time.time()

        if 'oof_pred' in result:
            oof_preds = result['oof_pred']
            oof_pred = _avg_preds(oof_preds, avg, X.index)
            oof_pred = _rename_pred(oof_pred, encoder, y.name)
            result['oof_pred'] = oof_pred

        if 'new_pred' in result:
            new_preds = result['new_pred']
            new_pred = _avg_preds(new_preds, avg, X_new.index)
            new_pred = _rename_pred(new_pred, encoder, y.name)
            result['new_pred'] = new_pred

        if 'importance' in result:
            importances = result['importance']
            importance = _concat_imp(importances)
            result['importance'] = importance

        if 'score' in result:
            scores = result['score']
            scores = ld2dl(scores)
            scores = pd.DataFrame(scores)
            result['score'] = scores

        for key in ['fit_time', 'score_time', 'pred_time']:
            if key in result:
                result[key] = np.array(result[key])

        concat_time = time.time() - start_time
        result['concat_time'] = concat_time

    result['use_cols'] = X.columns.copy()

    if not return_score and 'score' in result:
        result.pop('score')

    if not return_folds:
        result.pop('fold')

    if return_encoder:
        result['encoder'] = encoder


    # Final score
    logger.log_final(result)

    return result



def crossval_score(estimator, cv, X, y, groups=None, scoring=None,
                   n_jobs=-1, verbose=1, n_digits=4):
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


    Returns
    -------
    scores : Series or DataFrame
        Rows are splits. If multimetric, return DataFrame, where each column
        represents different metric.

    """
    result = crossval(estimator, cv, X, y, groups, n_digits=n_digits,
                      scoring=scoring, return_score=True, n_jobs=n_jobs,
                      verbose=verbose)

    scores = result['score']
    return scores



def crossval_predict(estimator, cv, X, y, groups=None, X_new=None,
                     test_avg=True, averaging='auto', method='predict',
                     scoring=None, n_jobs=-1, verbose=0, n_digits=4):
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

        Ignored if return_pred=False or X_new is not defined.

    averaging : string, {'soft', 'hard', 'auto', 'pass'} (default='auto')
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

        - 'pass' : leave predictions of different folds separated.

                   Column 'fold' will be added.

        Ignored if <return_pred> set to False, or <method> is not 'predict'.

    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.
        Ignored if return_pred=False.

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


    Returns
    -------
    oof_pred : Series, shape [n_samples]
        Out-of-fold predictions

    new_pred : Series, shape [m_samples] or None
        Test predictions (unseen data)
        None if X_new is not defined

    """
    result = crossval(estimator, cv, X, y, groups, X_new=X_new, scoring=scoring,
                      averaging=averaging, method=method, test_avg=test_avg,
                      return_estimator=False, return_pred=True,  return_score=False,
                      n_jobs=n_jobs, verbose=verbose, n_digits=n_digits)

    oof_pred = result['oof_pred'] if 'oof_pred' in result else None
    new_pred = result['new_pred'] if 'new_pred' in result else None

    return oof_pred, new_pred


def _pass_pred(pred):
    return pred

def _mean_pred(pred):
    pred.reset_index('fold', inplace=True, drop=True)
    return pred.groupby(pred.index).mean()

def _soft_vote(pred):
    pred.reset_index('fold', inplace=True, drop=True)
    return pred.groupby(pred.index).mean().idxmax(axis=1)

def _hard_vote(pred):
    pred.reset_index('fold', inplace=True, drop=True)
    return pred.idxmax(axis=1).groupby(pred.index).agg(pd.Series.mode)


def _check_avg(estimator, averaging, method):

    name = extract_model_name(estimator)

    # Basic <method> and <averaging> values check
    methods = ['predict', 'predict_proba']
    if method not in methods:
        raise ValueError("<method> should be in {}".format(set(methods)) \
            + "\n\t\tPassed '{}'".format(method))

    averagings = ['soft', 'hard', 'auto', 'pass']
    if averaging not in averagings:
        raise ValueError("<averaging> should be in {}".format(set(averagings)) \
            + "\n\t\tPassed '{}'".format(averaging))

    # Compatibility check
    if is_classifier(estimator) and hasattr(estimator, 'predict_proba'):
        # classifier (probabilistic)
        if method is 'predict_proba':

            if averaging is 'pass':
                avg = _pass_pred

            elif averaging is 'auto':
                avg = _mean_pred

            else:
                good_vals = {'auto', 'pass'}
                bad_vals = {'soft', 'hard'}
                msg = "Selected <method> value is {}.".format(method) \
                    + "\n\t\tAvailable <averaging> options are: {}.".format(good_vals) \
                    + "\n\t\tBut current <averaging> value set to '{}'.".format(averaging) \
                    + "\n\t\t" \
                    + "\n\t\tNote: {} are voting strategies, so".format(bad_vals) \
                    + "\n\t\tthey are available only for method='predict'."
                raise ValueError(msg)

        elif method is 'predict':

            method = 'predict_proba'

            if averaging in ['soft', 'auto']:
                avg = _soft_vote

            elif averaging is 'hard':
                avg = _hard_vote

            elif averaging is 'pass':
                avg = _pass_pred


    elif is_classifier(estimator) and not hasattr(estimator, 'predict_proba'):
        # classifier (non-probabilistic)
        if method is 'predict_proba':

            msg = "<{}> is not available for <{}>".format(method, name)
            raise AttributeError(msg)

        elif method is 'predict':

            if averaging in ['hard', 'auto']:
                avg = _hard_vote

            elif averaging is 'pass':
                avg = _pass_pred

            else:
                vals = {'auto', 'hard', 'pass'}
                msg = "<{}> is a {}. ".format(name, 'non-probabilistic classifier') \
                    + "\n\t\tAvailable <averaging> options are: {}".format(vals) \
                    + "\n\t\tCurrent value set to '{}'".format(averaging)
                raise ValueError(msg)

    elif is_regressor(estimator):
        # regressor
        if averaging is 'pass':
            avg = _pass_pred
            method = 'predict'

        elif averaging is 'auto':
            avg = _mean_pred
            method = 'predict'

        else:
            vals = {'auto', 'pass'}
            msg = "<{}> is a {}. ".format(name, 'regressor') \
                + "\n\t\tAvailable <averaging> options are: {}".format(vals) \
                + "\n\t\tCurrent value set to '{}'".format(averaging)
            raise ValueError(msg)

    # FIXME: Crashes on estimators without <_estimator_type> attribute.

    return method, avg


def _fit_pred_score(estimator, method, scorers, X, y, trn=None, oof=None, X_new=None,
                    return_pred=False, return_estimator=False, return_score=True,
                    return_importance=False, fold_ind=None, logger=None):
    """Fit estimator and evaluate metric(s), compute OOF predictions & etc.

    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data.

    X : DataFrame, shape [n_samples, n_features]
        The data to fit, score and calculate out-of-fold predictions

    y : Series, shape [n_samples]
        The target variable to try to predict

    trn : array or None
        Indices of rows, selected to fit estimator. If None, select all.

    oof : array or None
        Indices of rows, selected to score estimator. If None, select none.

    X_new : DataFrame, shape [m_samples, n_features] or None
        The unseed data to predict (test set)

    scorer : scorer object
        A scorer callable object with signature ``scorer(estimator, X, y)``
        which should return only a single value.

    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order. Ignored if return_pred=False.

    return_pred : bool (default=False)
        Return out-of-fold prediction (and test prediction, if X_new is defined)

    return_estimtor : bool (default=False)
        Return fitted estimator

    return_score : bool (default=True)
        Return out-of-fold score

    return_importance : bool (default=False)
        Return averaged <feature_importances_> or <coef_> of fitted estimator


    Returns
    -------
    result : dict of float or Series
        Scores/predictions/time of the estimator for each run of the
        cross validation. The possible keys for this ``dict`` are:

            ``fold`` : pair of list
                Two lists with trn/oof indices

            ``score`` : float
                The OOF score. If multimetric, return dict of float.
                Ignored if return_score=False.

            ``oof_pred`` : Series, shape [n_samples]
                OOF predictions. Ignored if return_pred=False.

            ``new_pred`` : Series, shape [m_samples]
                Test predictions (unseen data). Ignored if return_pred=False.

            ``fit_time`` : float
                The time for fitting the estimator

            ``pred_time`` : float
                OOF and test predictions time. Ignored if return_pred=False.

            ``score_time`` : float
                OOF score time. Ignored if return_score=False.

            ``estimator`` : estimator object
                The fitted estimator object. Ignored if return_estimator=False.

            ``importance`` : Series, shape [n_features]
                <feature_importances_> or <coef_> of fitted estimator

    """
    result = {}

    # Get indices
    trn = np.arange(len(X)) if trn is None else trn
    oof = np.arange(0) if oof is None else oof
    result['fold'] = (trn, oof)

    new = np.arange(len(X_new)) if X_new is not None else np.arange(0)

    # Split data
    start_time = time.time()

    X_trn, y_trn = _safe_split(estimator, X, y, trn)
    X_oof, y_oof = _safe_split(estimator, X, y, oof)

    # Fit estimator
    estimator.fit(X_trn, y_trn)

    fit_time = time.time() - start_time
    result['fit_time'] = fit_time

    if return_estimator:
        result['estimator'] = estimator

    # Feature importances
    if return_importance:
        importance = extract_importance(estimator, X)
        result['importance'] = importance

    # Predict
    if return_pred and (len(oof) or len(new)):

        start_time = time.time()

        if len(oof):
            oof_pred = _pred(estimator, method, X_oof, y.name)
            result['oof_pred'] = oof_pred

        if len(new):
            new_pred = _pred(estimator, method, X_new, y.name)
            result['new_pred'] = new_pred

        pred_time = time.time() - start_time
        result['pred_time'] = pred_time

    # Score
    if return_score and scorers and len(oof):

        start_time = time.time()

        scores = _score(estimator, X_oof, y_oof, scorers)
        result['score'] = scores

        score_time = time.time() - start_time
        result['score_time'] = score_time

    # Logging
    if logger:
        logger.log(fold_ind, result)

    return result


def _pred(estimator, method, X, target):
    """Call <method> of fitted <estimator> on data <X>.

    Parameters
    ----------
    estimator : estimator object
        Fitted estimator

    method : iterable of string
        Feature names

    X : DataFrame, shape [k_samples, k_features]
        The unseed data to predict

    target : string
        Name of target column


    Returns
    -------
    pred : Series or DataFrame, shape [n_features] or [n_features, n_classes]
        Computed predictions

    """

    # Check Attribute
    if hasattr(estimator, method):
        action = getattr(estimator, method)
    else:
        name = extract_model_name(estimator)
        raise AttributeError("<{}> has no method <{}>".format(name, method))

    # Predict
    pred = action(X)

    # Convert numpy to pandas
    if len(pred.shape) is 1:
        pred = pd.Series(pred, index=X.index, name=target)
    else:
        pred = pd.DataFrame(pred, index=X.index)

    return pred


def _score(estimator, X, y, scorers):

    scores = {}

    for name, scorer in scorers.items():

        if y is None:
            score = scorer(estimator, X)
        else:
            score = scorer(estimator, X, y)

        scores[name] = score

    return scores


def _avg_preds(preds, avg, ind):
    """Concatenate predictions, using <avg> function

    Parameters
    ----------
    preds : list of Series or DataFrame
        Estimators predictions by fold

    avg : callable
        Function, used to aggregate predictions (averaging or voting)

    ind : iterable
        Original indices order of data, used to compute predictions


    Returns
    -------
    pred : Series or DataFrame, shape [n_features] or [n_features, n_classes]
        Computed predictions

    """
    # Add fold index
    for i, pred in enumerate(preds):
        pred = pd.DataFrame(pred)
        pred.insert(0, 'fold', i)
        preds[i] = pred

    # Concatenate & sort
    pred = pd.concat(preds)
    pred = pred.loc[ind]

    # Average predictions
    pred = pred.set_index('fold', append=True)
    pred = avg(pred)

    return pred


def _rename_pred(pred, encoder, target):
    """Decode columns (or labels) back and rename target column

    Parameters
    ----------
    pred : Series or DataFrame
        Predictions

    encoder : transformer object or None
        Transformer, used to encode target labels. Must has <inverse_transform>
        method. If None, interpreted as non-classification task.

    target : string
        Name of target column


    Returns
    -------
    pred : Series or DataFrame, shape [n_features] or [n_features, n_classes]
        Computed predictions

    """

    pred = pd.DataFrame(pred)

    # Target Decoding
    if encoder:
        if len(pred.columns) is 1:
            # regression or non-probabilistic classification
            pred.columns = [target]
            pred[target] = encoder.inverse_transform(pred[target])
        else:
            # probabilistic classification
            pred.columns = encoder.inverse_transform(pred.columns)

    # Binary Classification
    if set(pred.columns) == {0, 1}:
        pred.drop(columns=0, inplace=True)
        pred.columns = [target]

    # Single Column DataFrame to Series
    if len(pred.columns) is 1:
        pred = pred.iloc[:,0]

    return pred


def _concat_imp(importances):
    """Concatenate importances

    Parameters
    ----------
    importances : list of Series
        Estimators feature importances by fold


    Returns
    -------
    importance : DataFrame, shape [n_features, 2]
        DataFrame with columns ['fold', 'importance'] and index 'feature'

    """
    importance = pd.DataFrame(importances).reset_index(drop=True)
    importance = importance.stack().reset_index()

    importance.columns = ['fold', 'feature', 'importance']
    importance.set_index(['feature','fold'], inplace=True)

    return importance
