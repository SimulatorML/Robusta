import pandas as pd
import numpy as np

from joblib import Parallel, delayed
import time

from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _score
from sklearn.model_selection._split import check_cv
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils import indexable

from ..preprocessing import LabelEncoder1D


__all__ = ['crossval', 'crossval_score', 'crossval_predict']




def crossval(estimator, cv, X, y, groups=None, X_new=None, test_avg=True,
             scoring=None, voting='auto', method='predict', return_pred=False,
             return_estimator=False, return_score=True, return_importance=False,
             return_encoder=False, n_jobs=-1, verbose=1):
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

    voting : string, {'soft', 'hard', 'auto'} (default='auto')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers. If 'auto', select 'soft'
        for estimators that has <predict_proba>, otherwise 'hard'.
        Ignored if return_pred=False or estimator type is not 'classifier'.

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

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int
        Verbosity level


    Returns
    -------
    results : dict of array, float or Series
        Array of scores/predictions/time of the estimator for each run of the
        cross validation. If test_avg=True, arrays has shape [n_splits],
        otherwise [n_splits+1] except score & score_time.

        The possible keys for this ``dict`` are:

            ``score`` : array or dict of array, shape [n_splits]
                The score array for test scores on each cv split.
                If multimetric, return dict of array.
                Ignored if return_score=False.

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
    folds = cv.split(X, y, groups)

    # Check scorer(s)
    scorer, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    # Check voting strategy & method
    method, avg = _check_voting(estimator, voting, method, return_pred)

    # Target Encoding
    if is_classifier(estimator):
        encoder = LabelEncoder1D()
        y = encoder.fit_transform(y)
    else:
        encoder = None

    # Fit & predict
    parallel = Parallel(max_nbytes='256M', pre_dispatch='2*n_jobs',
                        n_jobs=n_jobs)

    if test_avg:

        # Stacking Type A (test_averaging = True)
        results = parallel(
            (delayed(_fit_pred_score)(clone(estimator), method, scorer, X, y,
                trn, oof, X_new, return_pred, return_estimator, return_score,
                return_importance)
            for trn, oof in folds))

        results = _ld_to_dl(results)

    else:

        # Stacking Type B (test_averaging = False)
        results = parallel(
            (delayed(_fit_pred_score)(clone(estimator), method, scorer, X, y,
                trn, oof, None, return_pred, return_estimator, return_score,
                return_importance)
            for trn, oof in folds))

        result_new = _fit_pred_score(clone(estimator), method, None, X, y,
            None, None, X_new, return_pred, return_estimator, False,
            return_importance)

        results = _ld_to_dl(results)
        for key, val in result_new.items():
            if key in results:
                results[key].append(val)
            else:
                results[key] = [val]


    # Concat Predictions (& Feature Importances)
    needs_concat = ['oof_pred', 'new_pred', 'importance', 'score']
    if np.any(np.in1d(needs_concat, list(results))):

        start_time = time.time()

        if 'oof_pred' in results:
            oof_preds = results['oof_pred']
            oof_pred = _concat_preds(oof_preds, avg, encoder, y.name, X.index)
            results['oof_pred'] = oof_pred

        if 'new_pred' in results:
            new_preds = results['new_pred']
            new_pred = _concat_preds(new_preds, avg, encoder, y.name, X_new.index)
            results['new_pred'] = new_pred

        if 'importance' in results:
            importances = results['importance']
            importance = pd.DataFrame(importances).reset_index(drop=True)
            results['importance'] = importance

        if 'score' in results:
            scores = results['score']
            scores = _ld_to_dl(scores)
            scores = pd.DataFrame(scores)
            if scores.shape[1] == 1:
                scores = scores.iloc[:,0]
            results['score'] = scores

        for key in ['fit_time', 'score_time', 'pred_time']:
            if key in results:
                results[key] = np.array(results[key])

        concat_time = time.time() - start_time
        results['concat_time'] = concat_time

    # Save encoder
    if return_encoder:
        results['encoder'] = encoder

    return results



def crossval_score(estimator, cv, X, y, groups=None, scoring=None,
                   n_jobs=-1, verbose=1):
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

    verbose : int
        Verbosity level


    Returns
    -------
    scores : Series or DataFrame
        Rows are splits. If multimetric, return DataFrame, where each column
        represents different metric.

    """
    results = crossval(estimator, cv=cv, X=X, y=y, groups=groups,
        scoring=scoring, return_score=True, n_jobs=n_jobs, verbose=verbose)

    scores = results['score']
    return scores



def crossval_predict(estimator, cv, X, y, groups=None, X_new=None,
                     test_avg=True, voting='auto', method='predict',
                     n_jobs=-1, verbose=0):
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

    voting : string, {'soft', 'hard', 'auto'} (default='auto')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers. If 'auto', select 'soft'
        for estimators that has <predict_proba>, otherwise 'hard'.
        Ignored if return_pred=False or estimator type is not 'classifier'.

    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.
        Ignored if return_pred=False.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int
        Verbosity level


    Returns
    -------
    oof_pred : Series, shape [n_samples]
        Out-of-fold predictions

    new_pred : Series, shape [m_samples] or None
        Test predictions (unseen data)
        None if X_new is not defined

    """
    results = crossval(estimator, cv=cv, X=X, y=y, groups=groups, X_new=X_new,
        scoring=None, voting=voting, method=method, test_avg=test_avg,
        return_estimator=False, return_pred=True, return_score=False,
        n_jobs=n_jobs, verbose=verbose)

    oof_pred = results['oof_pred'] if 'oof_pred' in results else None
    new_pred = results['new_pred'] if 'new_pred' in results else None

    return oof_pred, new_pred



def _avg_preds(preds):
    pred = pd.concat(preds)
    return pred.groupby(pred.index).mean()



def _hard_vote(preds):
    pred = pd.concat(preds)
    return pred.groupby(pred.index).apply(lambda x: x.value_counts().index[0])
    # FIXME: old version crashes when there are more then one modes
    #return pred.groupby(pred.index).agg(pd.Series.mode)



def _soft_vote(preds):
    pred = pd.concat(preds)
    return pred.groupby(pred.index).mean().idxmax(axis=1)



def _check_voting(estimator, voting, method, return_pred=True):

    name = _extract_est_name(estimator)

    # Method & averaging strategy (voting type) check
    if return_pred:

        methods = ['predict', 'predict_proba', 'predict_log_proba']
        if method not in methods:
            raise ValueError("<method> should be in {}".format(set(methods)) \
                + "\n\t\tPassed '{}'".format(method))

        votings = ['soft', 'hard', 'auto']
        if voting not in votings:
            raise ValueError("<voting> should be in {}".format(set(votings)) \
                + "\n\t\tPassed '{}'".format(voting))

        if voting is 'auto':
            voting = 'soft' if hasattr(estimator, 'predict_proba') else 'hard'

        if is_classifier(estimator):

            if not hasattr(estimator, method):
                msg = "<{}> is not available for <{}>".format(method, name)
                raise AttributeError(msg)

            if not hasattr(estimator, 'predict_proba') and voting is 'soft':
                msg = "'{}' voting is not available for <{}> ".format(voting, name) \
                    + "\n\t\tIt has not <predict_proba> method"
                raise AttributeError(msg)

    # Method & averaging strategy selection
    if is_classifier(estimator):

        if method is 'predict_proba':
            avg = _avg_preds

        elif voting is 'hard':
            avg = _hard_vote

        elif voting is 'soft':
            avg = _soft_vote
            method = 'predict_proba'

        else:
            avg = _avg_preds
            method = 'predict'

    else:
        avg = _avg_preds
        method = 'predict'

    return method, avg



def _fit_pred_score(estimator, method, scorer, X, y, trn=None, oof=None, X_new=None,
                    return_pred=False, return_estimator=False, return_score=True,
                    return_importance=False):
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
        imporance = _imp(estimator, X.columns)
        result['importance'] = imporance

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
    if return_score and scorer and len(oof):

        start_time = time.time()

        is_multimetric = not callable(scorer)
        score = _score(estimator, X_oof, y_oof, scorer, is_multimetric)
        result['score'] = score

        score_time = time.time() - start_time
        result['score_time'] = score_time

    return result



def _imp(estimator, cols):
    """Extract <feature_importances_> or <coef_> from fitted estimator.

    Parameters
    ----------
    estimator : estimator object
        Fitted estimator

    cols : iterable of string
        Feature names


    Returns
    -------
    importance : Series, shape [n_features]
        Feature importances of fitted estimator

    """
    # Get importances
    if hasattr(estimator, 'coef_'):
        attr = 'coef_'
    elif hasattr(estimator, 'feature_importances_'):
        attr = 'feature_importances_'
    else:
        name = _extract_est_name(estimator)
        msg = "<{}> has neither <feature_importances_>, nor <coef_>".format(name)
        raise AttributeError(msg)

    imp = getattr(estimator, attr)

    # Convert numpy to pandas
    imp = pd.Series(imp, index=cols, name=attr)

    return imp



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
        name = _extract_est_name(estimator)
        raise AttributeError("<{}> has no method <{}>".format(name, method))

    # Predict
    pred = action(X)

    # Convert numpy to pandas
    if len(pred.shape) is 1:
        pred = pd.Series(pred, index=X.index, name=target)
    else:
        pred = pd.DataFrame(pred, index=X.index)

    return pred



def _concat_preds(preds, mean, encoder, name, ind):
    """Concatenate predictions, using <mean> function

    Parameters
    ----------
    preds : list of Series or DataFrame
        Fitted estimator

    mean : callable
        Function, which used to concatenate predictions

    encoder : transformer object or None
        Transformer, used to encode target labels. Must has <inverse_transform>
        method. If None, interpreted as non-classification task.

    name : string
        Name of target column

    ind : iterable
        Original indices order of data, used to compute predictions


    Returns
    -------
    pred : Series or DataFrame, shape [n_features] or [n_features, n_classes]
        Computed predictions

    """
    # Averaging predictions
    pred = mean(preds)

    # Target Decoding
    if encoder is not None:
        if len(pred.shape) is 1:
            pred = encoder.inverse_transform(pred)
            pred = pred.rename(name)
        else:
            pred.columns = encoder.inverse_transform(pred.columns)

    # Binary Classification
    if len(pred.shape) is 2 and set(pred.columns) == {0, 1}:
        pred = pred[1]
        pred = pred.rename(name)

    # Original indices order
    pred = pred.loc[ind]

    return pred



def _extract_est_name(estimator, drop_type=False):
    """Extract name of estimator instance.

    Parameters
    ----------
    estimator : estimator object
        Estimator or Pipeline

    drop_type : bool (default=False)
        Whether to remove an ending of the estimator's name, contains
        estimator's type. For example, 'XGBRegressor' transformed to 'XGB'.


    Returns
    -------
    name : string
        Name of the estimator

    """
    name = estimator.__class__.__name__

    if name is 'Pipeline':
        last_step = estimator.steps[-1][1]
        name = _extract_est_name(last_step, drop_type=drop_type)

    elif name is 'TransformedTargetRegressor':
        regressor = estimator.regressor
        name = _extract_est_name(regressor, drop_type=drop_type)

    elif drop_type:
        for etype in ['Regressor', 'Classifier', 'Ranker']:
            if name.endswith(etype):
                name = name[:-len(etype)]

    return name



def _ld_to_dl(l):
    return {key: [d[key] for d in l] for key in l[0].keys()}
