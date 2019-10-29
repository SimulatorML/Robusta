import pandas as pd
import numpy as np
import gc

from time import time

from sklearn.utils.metaestimators import _safe_split
from sklearn.base import is_classifier, is_regressor

from robusta.importance import get_importance
from scipy.stats import mode


__all__ = [
    '_fit_predict',
    '_predict',
    '_check_avg',
    '_avg_preds',
]



def _fit_predict(estimator, method, scorer, X, y, X_new=None, new_index=None,
                 trn=None, oof=None, return_estimator=False, return_pred=False,
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

    idx : int
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

            ``estimator`` : estimator object
                The fitted estimator object.
                Ignored if return_estimator=False.

            ``importance`` : Series, shape [n_features]
                Extracted feature importances

    """
    result = {}

    # Data
    new = np.arange(X_new.shape[0]) if X_new is not None else np.arange(0)
    trn = np.arange(X.shape[0]) if trn is None else trn
    oof = np.arange(0) if oof is None else oof

    X_trn, y_trn = _safe_split(estimator, X, y, trn)
    X_oof, y_oof = _safe_split(estimator, X, y, oof)

    oof_index = getattr(X_oof, 'index', y_oof.index)
    new_index = getattr(X_new, 'index', new_index)

    # Estimator
    tic = time()
    estimator.fit(X_trn, y_trn)
    result['fit_time'] = time() - tic

    if return_estimator:
        result['estimator'] = estimator

    # Feature Importances
    try:
        result['importance'] = get_importance(estimator)
    except:
        pass

    # Predict
    if return_pred and (len(oof) or len(new)):

        tic = time()

        if len(oof):
            oof_pred = _predict(estimator, method, X_oof, y, oof_index)
            result['oof_pred'] = oof_pred

        if len(new):
            new_pred = _predict(estimator, method, X_new, y, new_index)
            result['new_pred'] = new_pred

        result['pred_time'] = time() - tic

    # Score
    if scorer and len(oof):
        tic = time()
        result['score'] = scorer(estimator, X_oof, y_oof)
        result['score_time'] = time() - tic

    # Logs
    if logger:
        logger.log(fold_idx, result)

    return result



def _predict(estimator, method, X, y, index):
    """Call <method> of fitted <estimator> on data <X>.

    Parameters
    ----------
    estimator : estimator object
        Fitted estimator

    method : iterable of string
        Feature names

    X : DataFrame or 2d-array
        Data to predict

    y : string
        Target (used for prediction formatting).
        Could be already seen.

    index : iterable
        X indices (used for prediction formatting).

    Returns
    -------
    P : Series or DataFrame
        Computed predictions

    """
    Y = pd.DataFrame(y)

    name = estimator.__class__.__name__

    # Call method
    action = getattr(estimator, method, None)
    if action:
        P = action(X)
    else:
        raise AttributeError("<{}> has no method <{}>".format(name, method))

    # Format
    index = getattr(X, 'index', index)

    if isinstance(P, list):
        P = [pd.DataFrame(p, index=index) for p in P]
    else:
        P = [pd.DataFrame(P, index=index)]


    P = pd.concat(P, axis=1)

    if method is 'predict':
        # [classifier + predict] OR [regressor]
        P.columns = Y.columns

    elif is_classifier(estimator):
        # [classifier + predict_proba]
        if name in ['MultiOutputClassifier', 'MultiTargetClassifier']:
            # Multiple output classifier
            Classes = estimator.classes_

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



def _check_avg(estimator, avg_type, method):

    name = estimator.__class__.__name__

    # Basic <method> and <avg_type> values check
    methods = ['predict', 'predict_proba']
    if method not in methods:
        raise ValueError("<method> should be in {}".format(set(methods)) \
            + "\n\t\tPassed '{}'".format(method))

    avg_types = ['mean', 'soft', 'hard', 'rank', 'auto', 'pass']
    if avg_type not in avg_types:
        raise ValueError("<avg_type> should be in {}".format(set(avg_types)) \
            + "\n\t\tPassed '{}'".format(avg_type))

    # Compatibility check
    if is_classifier(estimator) and hasattr(estimator, 'predict_proba'):
        # classifier (probabilistic)
        if method is 'predict_proba':

            if avg_type is 'pass':
                avg = _pass_pred

            elif avg_type is 'rank':
                avg = _rank_pred

            elif avg_type in ['auto', 'mean']:
                avg = _mean_pred

            else:
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

            if avg_type in ['soft', 'auto']:
                method = 'predict_proba'
                avg = _soft_vote

            elif avg_type is 'hard':
                avg = _hard_vote

            elif avg_type is 'pass':
                avg = _pass_pred

            else:
                raise ValueError("Passed unavailable avg_type '{}' for method <{}>"
                                 "".format(avg_type, method))


        elif method is 'decision_function':

            if avg_type in ['mean', 'auto']:
                avg = _mean_pred

            elif avg_type is 'rank':
                avg = _rank_pred

            elif avg_type is 'pass':
                avg = _pass_pred

            else:
                raise ValueError("Passed unavailable avg_type '{}' for method <{}>"
                                 "".format(avg_type, method))


    elif is_classifier(estimator) and not hasattr(estimator, 'predict_proba'):
        # classifier (non-probabilistic)
        if method is 'predict_proba':

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
        # regressor
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



def _avg_preds(preds, avg, X, y, index):

    # Concat & sort
    index = getattr(X, 'index', index)

    pred = pd.concat(preds, axis=1)
    pred = pred.loc[index]

    del preds
    gc.collect()

    # Average
    pred = avg(pred)

    # Simplify
    if hasattr(pred, 'columns') and pred.shape[1] == 1:
        pred = pred.iloc[:, 0] # regression
    pred = _drop_zero_class(pred, y) # binary classifier

    return pred



def _drop_zero_class(pred, y):

    # Check if task is not regression
    if len(pred.shape) < 2:
        return pred

    # Check if avg_type is not 'pass'
    if (pred.columns.value_counts() > 1).any():
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



def _pass_pred(pred):
    return pred



def _mean_pred(pred):
    if hasattr(pred.columns, 'levels'):
        return _multioutput_vote(pred, _mean_pred)
    else:
        return pred.groupby(pred.columns, axis=1).mean()



def _rank_pred(pred):
    if hasattr(pred.columns, 'levels'):
        return _multioutput_vote(pred, _rank_pred)
    else:
        return pred.rank(pct=True).groupby(pred.columns, axis=1).mean()



def _soft_vote(pred):
    if hasattr(pred.columns, 'levels'):
        return _multioutput_vote(pred, _soft_vote)
    else:
        return pred.groupby(pred.columns, axis=1).mean().idxmax(axis=1)



def _hard_vote(pred):
    if hasattr(pred.columns, 'levels'):
        return _multioutput_vote(pred, _hard_vote)
    else:
        return pred.apply(lambda x: mode(x)[0][0], axis=1)



def _multioutput_vote(pred, vote):
    targets = pred.columns.get_level_values(0).unique()
    preds = [pred.loc[:, target] for target in targets]
    preds = [vote(p) for p in preds]
    pred = pd.concat(preds, axis=1)
    pred.columns = targets
    pred.columns.name = None
    return pred



def _multioutput_vote(pred, vote):
    targets = pred.columns.get_level_values(0).unique()
    preds = [pred.loc[:, target] for target in targets]
    preds = [vote(p) for p in preds]
    pred = pd.concat(preds, axis=1)
    pred.columns = targets
    pred.columns.name = None
    return pred
