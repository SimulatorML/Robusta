import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection._split import check_cv
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils import indexable

from ..preprocessing import LabelEncoder1D


__all__ = ['cross_val_stack']




def cross_val_stack(estimator, cv, X, y, groups=None, X_new=None,
                    test_average=True, voting='auto', method='predict',
                    n_jobs=None, pre_dispatch='2*n_jobs', verbose=0):

    # Check data
    X, y, groups = indexable(X, y, groups)
    X_new, _ = indexable(X_new, None)

    # Target Encoding
    if is_classifier(estimator):
        encoder = LabelEncoder1D()
        y = encoder.fit_transform(y)
    else:
        encoder = None

    # Init cross-validation scheme
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    folds = cv.split(X, y, groups)

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

    # Fit & predict
    if test_average:

        # Variant A:
        jobs = (delayed(_fit_pred)(clone(estimator), method, X, y, trn, oof, X_new)
                                   for trn, oof in folds)
        paths = Parallel(backend='multiprocessing', max_nbytes='256M', n_jobs=n_jobs,
                         verbose=verbose, pre_dispatch=pre_dispatch)(jobs)

        oof_preds, new_preds = zip(*paths)

    else:

        # Variant B
        jobs = (delayed(_fit_pred)(clone(estimator), method, X, y, trn, oof)
                                   for trn, oof in folds)
        paths = Parallel(backend='multiprocessing', max_nbytes='256M', n_jobs=n_jobs,
                         verbose=verbose, pre_dispatch=pre_dispatch)(jobs)

        oof_preds, _ = zip(*paths)

        _, new_pred = _fit_pred(clone(estimator), method, X, y, X_new=X_new)
        new_preds = [new_pred]

    # Concat Predictions
    oof_pred = _concat_preds(oof_preds, mean, X, y, encoder)
    new_pred = _concat_preds(new_preds, mean, X_new, y, encoder)

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



def _fit_pred(estimator, method, X, y, trn=None, oof=None, X_new=None):

    # Get indices
    trn = np.arange(len(X)) if trn is None else trn
    oof = np.arange(0) if oof is None else oof

    new = np.arange(len(X_new)) if X_new is not None else np.arange(0)

    # Split data
    X_trn, y_trn = _safe_split(estimator, X, y, trn)
    X_oof, y_oof = _safe_split(estimator, X, y, oof)

    # Fit
    estimator.fit(X_trn, y_trn)

    # Predict
    oof_pred = _pred(estimator, method, X_oof, y) if len(oof) else None
    new_pred = _pred(estimator, method, X_new, y) if len(new) else None

    return oof_pred, new_pred



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



def _concat_preds(preds, mean, X, y, encoder=None):

    # Check if empty
    print(X)
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
