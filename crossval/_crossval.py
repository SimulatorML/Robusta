import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection._split import check_cv
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import check_scoring
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils import indexable

from ..preprocessing import LabelEncoder1D


__all__ = ['cross_val_stack']




def cross_val_stack(estimator, X, y, groups=None, cv='warn', X_new=None,
                    test_average=True, voting='auto', method='predict',
                    n_jobs=None, pre_dispatch='2*n_jobs', verbose=0):

    # Check data
    X, y, groups = indexable(X, y, groups)
    X_new, _ = indexable(X_new, None)

    # Label Encoding
    le = LabelEncoder1D()
    y = le.fit_transform(y)

    # Init cross-validation scheme
    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    # Method & averaging strategy check
    methods = ['predict', 'predict_proba']
    if method not in methods:
        raise ValueError('<method> should be in {}'.format(set(methods)))

    votings = ['soft', 'hard', 'auto']
    if voting not in votings:
        raise ValueError('<voting> should be in {}'.format(set(votings)))

    if voting is 'auto':
        voting = 'soft' if hasattr(estimator, 'predict_proba') else 'hard'

    if not hasattr(estimator, 'predict_proba') and is_classifier(estimator):
        if method is 'predict_proba' or voting is 'soft':
            msg = "<predict_proba> is not available for this estimator" + \
                "\nSet <method> to 'predict' and <voting> to 'hard' or 'auto'"
            raise AttributeError(msg)

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

    # Fit & predict for each split
    jobs = (delayed(_fit_pred)(clone(estimator), method, X, y, trn, oof, X_new)
                               for trn, oof in cv.split(X, y, groups))
    paths = Parallel(backend='multiprocessing', max_nbytes='256M', n_jobs=n_jobs,
                     verbose=verbose, pre_dispatch=pre_dispatch)(jobs)

    # Averaging predictions
    oof_preds, new_preds = zip(*paths)

    oof_pred = mean(oof_preds)
    new_pred = mean(new_preds)

    # Label Decoding
    if len(oof_pred.shape) is 1:
        oof_pred = le.inverse_transform(oof_pred).rename(y.name)
        new_pred = le.inverse_transform(new_pred).rename(y.name)
    else:
        oof_pred.columns = le.inverse_transform(oof_pred.columns)
        new_pred.columns = le.inverse_transform(new_pred.columns)

    # Binary Classification
    if len(oof_pred.shape) is 2 and set(oof_pred.columns) == {0, 1}:
        oof_pred = oof_pred[1].rename(y.name)
        new_pred = new_pred[1].rename(y.name)

    # Original indices order
    oof_pred = oof_pred.loc[X.index]
    new_pred = new_pred.loc[X_new.index]

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



def _fit_pred(estimator, method, X, y, trn, oof, X_new):

    X_trn, y_trn = _safe_split(estimator, X, y, trn)
    X_oof, y_oof = _safe_split(estimator, X, y, oof)

    estimator.fit(X_trn, y_trn)

    if hasattr(estimator, method):
        func = getattr(estimator, method)

        oof_pred = func(X_oof)
        new_pred = func(X_new)

        if len(oof_pred.shape) is 1:
            oof_pred = pd.Series(oof_pred, index=X_oof.index, name=y.name)
            new_pred = pd.Series(new_pred, index=X_new.index, name=y.name)
        else:
            oof_pred = pd.DataFrame(oof_pred, index=X_oof.index)
            new_pred = pd.DataFrame(new_pred, index=X_new.index)

        return oof_pred, new_pred

    else:
        name = estimator.__class__.__name__
        raise AttributeError("<{}> has no method <{}>".format(name, method))
