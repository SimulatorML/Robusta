import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from time import time
import datetime

from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring
from sklearn.utils import indexable

from robusta.preprocessing import LabelEncoder1D, LabelEncoder
from robusta.model import extract_model_name, extract_model, get_model_name
from robusta.importance import get_importance, permutation_importance
from robusta.utils import logmsg, ld2dl

from ._output import CVLogger


__all__ = ['crossval', 'crossval_score', 'crossval_predict']




def crossval(estimator, cv, X, y, groups=None, X_new=None, test_avg=True,
             scoring=None, averaging='auto', method='predict', return_pred=True,
             return_estimator=True, return_importance=False, random_state=0,
             verbose=2, n_digits=4, n_jobs=-1):
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

                   Column '_FOLD' will be added.

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

    return_importance : bool (default=False)
        Return feature importances

    random_state : int or None, optional (default=0)
        Random seed for cross-validation split

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
    if hasattr(cv, 'random_state') and cv.random_state is None:
        cv.random_state = random_state

    avg, method = _check_avg(estimator, averaging, method)
    scorer = check_scoring(estimator, scoring)

    # Fit & predict
    logger = CVLogger(cv, verbose, prec=n_digits)
    logger.log_start(estimator, scorer)

    parallel = Parallel(max_nbytes='256M', pre_dispatch='2*n_jobs',
                        n_jobs=n_jobs, require='sharedmem')

    if test_avg:

        # Stacking Type A (test averaging = True)
        result = parallel(
            delayed(_fit_pred_score)(
                clone(estimator), method, scorer, X, y, X_new, trn, oof,
                return_pred, return_estimator, return_importance, i, logger)
            for i, (trn, oof) in enumerate(cv.split(X, y, groups)))

        result = ld2dl(result)

    else:

        # Stacking Type B (test_averaging = False)
        result = parallel(
            (delayed(_fit_pred_score)(
                clone(estimator), method, scorer, X, y, None, trn, oof,
                return_pred, return_estimator, return_importance, i, logger)
            for i, (trn, oof) in enumerate(cv.split(X, y, groups))))

        if verbose >= 2:
            print()
            logmsg('Fitting full train set...')

        result_new = _fit_pred_score(clone(estimator), method, None, X, y, X_new,
            None, None, return_pred, return_estimator, False, -1, None)

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
            oof_pred = _avg_preds(oof_preds, avg, X.index)
            oof_pred = _short_binary(oof_pred, y, averaging)
            result['oof_pred'] = oof_pred

        if 'new_pred' in result:
            new_preds = result['new_pred']
            new_pred = _avg_preds(new_preds, avg, X_new.index)
            new_pred = _short_binary(new_pred, y, averaging)
            result['new_pred'] = new_pred

        if 'importance' in result:
            importance = np.array(result['importance']).T
            importance = pd.DataFrame(importance, index=X.columns)
            result['importance'] = importance

        for key in ['fit_time', 'score_time', 'pred_time']:
            if key in result:
                result[key] = np.array(result[key])

        result['concat_time'] = time() - tic

    result['datetime'] = datetime.date.today()
    result['features'] = list(X.columns)
    result['cv'] = cv

    # Final score
    logger.log_final(result)

    return result



def crossval_score(estimator, cv, X, y, groups=None, scoring=None, n_jobs=-1,
                   verbose=1, n_digits=4):
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

    averaging : string, {'soft', 'hard', 'auto', 'pass'} (default='auto')
        Averaging strategy for aggregating different CV folds predictions

        - 'hard' : use predicted class labels for majority rule voting.

                   Ignored if estimator type is 'regressor'.
                   Ignored if <method> is not 'predict'.

        - 'soft' : predicts the class label based on the argmax of the sums
                   of the predicted probabilities, which is recommended for
                   an ensemble of well-calibrated classifiers.

                   Ignored if estimator type is 'regressor'.
                   Ignored if <method> is not 'predict'.

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
    pred = pred.copy()
    pred.reset_index('_FOLD', inplace=True, drop=True)
    return pred.groupby(pred.index).mean()


def _soft_vote(pred):
    pred = pred.copy()
    if hasattr(pred.columns, 'levels'):
        return _multioutput_vote(pred, _soft_vote)
    else:
        pred.reset_index('_FOLD', inplace=True, drop=True)
        return pred.groupby(pred.index).mean().idxmax(axis=1)


def _hard_vote(pred):
    pred = pred.copy()
    if hasattr(pred.columns, 'levels'):
        return _multioutput_vote(pred, _hard_vote)
    else:
        pred.reset_index('_FOLD', inplace=True, drop=True)
        return pred.idxmax(axis=1).groupby(pred.index).agg(lambda x: pd.Series.mode(x)[0])


def _multioutput_vote(pred, vote):
    targets = pred.columns.get_level_values(0).unique()
    preds = [pred.loc[:, target] for target in targets]
    preds = [vote(p) for p in preds]
    pred = pd.concat(preds, axis=1)
    pred.columns = targets
    pred.columns.name = None
    return pred


def _short_binary(pred, y, averaging):
    if averaging is 'pass':
        return pred

    if len(pred.shape) < 2:
        return pred

    if hasattr(pred.columns, 'levels'):
        targets = pred.columns.get_level_values(0).unique()
        preds = [pred.loc[:, target] for target in targets]
        is_binary = [list(p.columns) == [0, 1] for p in preds]
        is_binary = np.array(is_binary).all()

        if is_binary:
            preds = [p.loc[:, 1] for p in preds]
            pred = pd.concat(preds, axis=1)
            pred.columns = targets
            pred.columns.name = None

    elif list(pred.columns) == [0, 1]:
        pred = pred.loc[:, 1]
        pred.name = y.name

    return pred



def _check_avg(estimator, averaging, method):

    estimator = extract_model(estimator)
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

    return avg, method


def _fit_pred_score(estimator, method, scorer, X, y, X_new=None, trn=None, oof=None,
                    return_pred=False, return_estimator=False, return_importance=False,
                    fold_idx=None, logger=None):
    """Fit estimator and evaluate metric(s), compute OOF predictions & etc.

    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data.

    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order. Ignored if return_pred=False.

    scorer : scorer object
        A scorer callable object with signature ``scorer(estimator, X, y)``
        which should return only a single value.

    X : DataFrame, shape [n_samples, n_features]
        The data to fit, score and calculate out-of-fold predictions

    y : Series, shape [n_samples]
        The target variable to try to predict

    X_new : DataFrame, shape [m_samples, n_features] or None
        The unseed data to predict (test set)

    trn : array or None
        Indices of rows, selected to fit estimator. If None, select all.

    oof : array or None
        Indices of rows, selected to score estimator. If None, select none.

    return_pred : bool (default=False)
        Return out-of-fold prediction (and test prediction, if X_new is defined)

    return_estimator : bool (default=False)
        Return fitted estimator

    return_importance : bool (default=False)
        Return feature importances

    fold_idx : int
        Fold index. -1 for full dataset.

    logger : object
        Logger object

    Returns
    -------
    result : dict of float or Series
        Scores/predictions/time of the estimator for each run of the
        cross validation. The possible keys for this ``dict`` are:

            ``score`` : float
                The OOF score. If multimetric, return dict of float.

            ``oof_pred`` : Series, shape [n_samples]
                OOF predictions.
                Ignored if return_pred=False.

            ``new_pred`` : Series, shape [m_samples]
                Test predictions (unseen data).
                Ignored if return_pred=False.

            ``fit_time`` : float
                The time for fitting the estimator

            ``pred_time`` : float
                OOF and test predictions time.
                Ignored if return_pred=False.

            ``score_time`` : float
                OOF score time.

            ``importance_time`` : float
                Feature Importance extraction time.
                Ignored if imp=None.

            ``estimator`` : estimator object
                The fitted estimator object.
                Ignored if return_estimator=False.

            ``importance`` : Series, shape [n_features]
                Extracted feature importances

    """
    result = {}

    # Split data
    new = np.arange(len(X_new)) if X_new is not None else np.arange(0)
    trn = np.arange(len(X)) if trn is None else trn
    oof = np.arange(0) if oof is None else oof

    X_trn, y_trn = X.iloc[trn], y.iloc[trn]
    X_oof, y_oof = X.iloc[oof], y.iloc[oof]

    # Fit estimator
    tic = time()
    estimator.fit(X_trn, y_trn)
    result['fit_time'] = time() - tic

    if return_estimator:
        result['estimator'] = estimator

    # Feature importances
    if return_importance:
        result['importance'] = get_importance(estimator)

    # Predict
    if return_pred and (len(oof) or len(new)):

        tic = time()

        if len(oof):
            oof_pred = _pred(estimator, method, X_oof, y)
            result['oof_pred'] = oof_pred

        if len(new):
            new_pred = _pred(estimator, method, X_new, y)
            result['new_pred'] = new_pred

        result['pred_time'] = time() - tic

    # Score
    if scorer and len(oof):
        tic = time()
        result['score'] = scorer(estimator, X_oof, y_oof)
        result['score_time'] = time() - tic

    # Logging
    if logger:
        logger.log(fold_idx, result)

    return result


def _pred(estimator, method, X, Y):
    """Call <method> of fitted <estimator> on data <X>.

    Parameters
    ----------
    estimator : estimator object
        Fitted estimator

    method : iterable of string
        Feature names

    X : DataFrame, shape [k_samples, k_features]
        The unseed data to predict

    Y : string
        The unseed target (format)


    Returns
    -------
    P : Series or DataFrame
        Computed predictions

    """
    Y = pd.DataFrame(Y)

    # Check Attribute
    if hasattr(estimator, method):
        action = getattr(estimator, method)
    else:
        name = extract_model_name(estimator)
        raise AttributeError("<{}> has no method <{}>".format(name, method))

    # Predict
    P = action(X)

    # Format
    if isinstance(P, list):
        P = [pd.DataFrame(p, index=X.index) for p in P]
    else:
        P = [pd.DataFrame(P, index=X.index)]

    P = pd.concat(P, axis=1)

    if method is 'predict':
        # [classifier + predict] OR [regressor]
        P.columns = Y.columns

    elif is_classifier(estimator):
        # [classifier + predict_proba]
        name = get_model_name(estimator)

        if name in ['MultiOutputClassifier', 'MultiTargetClassifier']:
            # Multiple output classifier
            Classes = [e.classes_ for e in estimator.estimators_]

            tuples = []
            for target, classes in zip(Y.columns, Classes):
                for c in classes:
                    tuples.append((target, c))

            P.columns = pd.MultiIndex.from_tuples(tuples, names=('_TARGET', '_CLASS'))

        elif hasattr(estimator, 'classes_'):
            # Single output classifier
            classes = estimator.classes_
            P.columns = classes

        else:
            # Unknown classifier
            msg = "Classifier <{}> should has <classes_> attribute!".format(name)
            raise AttributeError(msg)

    else:
        # Ranker & etc
        est_type = getattr(estimator, "_estimator_type", None)
        raise TypeError('<{}> is an estimator of unknown type: \
                         <{}>'.format(name, est_type))

    return P


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
        pred.insert(0, '_FOLD', i)
        preds[i] = pred

    # Concatenate & sort
    pred = pd.concat(preds)
    pred = pred.loc[ind]

    # Average predictions
    pred = pred.set_index('_FOLD', append=True)
    pred = avg(pred)

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
        DataFrame with columns ['_FOLD', 'importance'] and index 'feature'

    """
    importance = pd.DataFrame(importances)#.reset_index(drop=True)
    #importance = importance.stack().reset_index()

    #importance.columns = ['_FOLD', 'feature', 'importance']
    #importance.set_index(['feature','_FOLD'], inplace=True)

    return importance
