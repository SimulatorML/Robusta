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


__all__ = ['cross_val', 'cross_val_stack']




def cross_val(estimator, cv, X, y, groups=None, X_new=None, test_avg=True,
              scoring=None, voting='auto', method='predict', return_pred=False,
              return_estimator=False, return_score=True, return_importance=True,
              n_jobs=None, pre_dispatch='2*n_jobs', verbose=0):

    # Check data
    X, y, groups = indexable(X, y, groups)
    X_new, _ = indexable(X_new, None)

    # Check validation scheme & scorer(s)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    folds = cv.split(X, y, groups)

    scorer, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    # Target Encoding
    if is_classifier(estimator):
        encoder = LabelEncoder1D()
        y = encoder.fit_transform(y)
    else:
        encoder = None

    # Method & averaging strategy (voting type) check
    if return_pred:

        methods = ['predict', 'predict_proba', 'predict_log_proba']
        if method not in methods:
            raise ValueError('<method> should be in {}'.format(set(methods)))

        votings = ['soft', 'hard', 'auto']
        if voting not in votings:
            raise ValueError('<voting> should be in {}'.format(set(votings)))

        if voting is 'auto':
            voting = 'soft' if hasattr(estimator, 'predict_proba') else 'hard'

        if is_classifier(estimator):

            if not hasattr(estimator, method):
                raise AttributeError("<{}> is not available for this estimator".format(method))

            if not hasattr(estimator, 'predict_proba') and voting is 'soft':
                raise AttributeError("'{}' voting is not available for this estimator \
                    cause it has not <{}> method".format(voting, method))

    # Method & averaging strategy selection
    if is_classifier(estimator):

        if method is 'predict_proba':
            mean = mean_pred

        elif voting is 'hard':
            mean = hard_vote

        elif voting is 'soft':
            mean = soft_vote
            method = 'predict_proba'

    else:
        mean = mean_pred
        method = 'predict'

    # Fit & predict
    parallel = Parallel(backend='multiprocessing', max_nbytes='256M', n_jobs=n_jobs,
                        verbose=verbose, pre_dispatch=pre_dispatch)

    if not return_pred or X_new is None or test_avg:

        # Variant A:
        results = parallel(
            (delayed(_fit_pred_score)(clone(estimator), method, scorer, X, y,
                trn, oof, X_new, return_estimator, return_score, return_pred)
            for trn, oof in folds))

    '''else:

        # Variant B
        jobs = (delayed(_fit_pred_score)(clone(estimator), method, scorer,
                                         X, y, trn, oof, None)
                                         for trn, oof in folds)
        paths = parallel(jobs)

        oof_preds, _ = zip(*paths)

        _, new_pred = _fit_pred_score(clone(estimator), method, None,
                                      X, y, None, None, X_new)
        new_preds = [new_pred]'''

    results = _ld_to_dl(results) # FIXME: list of dict -> dict of list

    # Concat Predictions
    if 'oof_pred' in results or 'new_pred' in results:

        start_time = time.time()

        if 'oof_pred' in results:
            oof_preds = results['oof_pred']
            oof_pred = _concat_preds(oof_preds, mean, encoder, X, y)
            results['oof_pred'] = oof_pred

        if 'new_pred' in results:
            new_preds = results['new_pred']
            new_pred = _concat_preds(new_preds, mean, encoder, X_new, y)
            results['new_pred'] = new_pred

        concat_time = time.time() - start_time
        results['concat_time'] = concat_time

    return results


def cross_val_stack(estimator, cv, X, y, groups=None, X_new=None,
                    test_avg=True, voting='auto', method='predict',
                    n_jobs=None, pre_dispatch='2*n_jobs', verbose=0):

    results = cross_val(estimator, cv, X, y, groups, X_new, test_avg, None,
        voting, method, True, False, False, False, n_jobs, pre_dispatch, verbose)
        # FIXME: positional args are not robust to <cross_val> args update

    oof_pred = results['oof_pred'] if 'oof_pred' in results else None
    new_pred = results['new_pred'] if 'new_pred' in results else None

    return oof_pred, new_pred



def mean_pred(preds):
    pred = pd.concat(preds)
    return pred.groupby(pred.index).mean()



def hard_vote(preds):
    pred = pd.concat(preds)
    return pred.groupby(pred.index).agg(pd.Series.mode)



def soft_vote(preds):
    pred = pd.concat(preds)
    return pred.groupby(pred.index).mean().idxmax(axis=1)



def _fit_pred_score(estimator, method, scorer, X, y, trn=None, oof=None, X_new=None,
                    return_estimator=False, return_score=True, return_pred=False):

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

    # Predict
    if return_pred and (len(oof) or len(new)):

        start_time = time.time()

        if len(oof):
            oof_pred = _pred(estimator, method, X_oof, y)
            result['oof_pred'] = oof_pred

        if len(new):
            new_pred = _pred(estimator, method, X_new, y)
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



def _pred(estimator, method, X, y):

    # Check Attribute
    if hasattr(estimator, method):
        action = getattr(estimator, method)
    else:
        name = estimator.__class__.__name__
        raise AttributeError("<{}> has no method <{}>".format(name, method))

    # Predict
    pred = action(X)

    # Convert numpy to pandas
    if len(pred.shape) is 1:
        pred = pd.Series(pred, index=X.index, name=y.name)
    else:
        pred = pd.DataFrame(pred, index=X.index)

    return pred



def _concat_preds(preds, mean, encoder, X, y):

    # Check if empty
    if X is None or X.empty:
        return

    # Averaging predictions
    pred = mean(preds)

    # Target Decoding
    if encoder is not None:
        if len(pred.shape) is 1:
            pred = encoder.inverse_transform(pred).rename(y.name)
        else:
            pred.columns = encoder.inverse_transform(pred.columns)

    # Binary Classification
    if len(pred.shape) is 2 and set(pred.columns) == {0, 1}:
        pred = pred[1].rename(y.name)

    # Original indices order
    pred = pred.loc[X.index]

    return pred



def _ld_to_dl(l):
    return {key: [d[key] for d in l] for key in l[0].keys()}
