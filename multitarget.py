import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from collections.abc import Iterable

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.base import clone, is_regressor, is_classifier
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import check_scoring


__all__ = [
    'MultiTargetRegressor',
    'MultiTargetClassifier',
]





class MultiTargetRegressor(BaseEstimator, RegressorMixin):
    """Multi target regression

    This strategy consists of fitting one regressor per target. This is a
    simple strategy for extending regressors that do not natively support
    multi-target regression.

    You can use either single estimator, either list of estimators.

    Parameters
    ----------
    estimator : estimator object, or list of estimators, shape (n_targets, )
        An estimator object implementing <fit> and <predict>.
        Or a list of estimators.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for <fit>. None means 1.
        ``-1`` means using all processors.

        When individual estimators are fast to train or predict
        using `n_jobs>1` can result in slower performance due
        to the overhead of spawning processes.

    """
    def __init__(self, estimator, scoring=None, weights=None, n_jobs=None):
        self.estimator = estimator
        self.scoring = scoring
        self.weights = weights
        self.n_jobs = n_jobs


    def fit(self, X, Y, sample_weight=None):
        """Fit the model to data.

        Fit a separate model for each output variable.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
        Y : DataFrame, shape (n_samples, n_targets)

        sample_weight : array-like, shape = (n_samples) or None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        Returns
        -------
        self : object

        """
        self.targets_ = list(Y.columns)
        self.estimators_ = check_estimator(self.estimator, self.targets_, 'regressor')

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(clone(e), X, Y[target])
            for e, target in zip(self.estimators_, self.targets_))

        return self

    def score(self, X, Y, *args, **kwargs):
        scores = []
        for estimator, target in zip(self.estimators_, self.targets_):
            scorer = check_scoring(estimator, self.scoring)
            score = scorer(estimator, X, Y[target], *args, **kwargs)
            scores.append(score)
        return np.average(scores, weights=self.weights)

    @property
    def feature_importances_(self):
        imps = [e.feature_importances_ for e in self.estimators_]
        return np.concatenate(imps).mean(axis=0)

    @property
    def coef_(self):
        imps = [e.coef_ for e in self.estimators_]
        return np.concatenate(imps).mean(axis=0)

    def predict(self, X):
        return _call_estimator(self, X, 'predict')




class MultiTargetClassifier(BaseEstimator, ClassifierMixin):
    """Multi target classification

    This strategy consists of fitting one classifier per target. This is a
    simple strategy for extending regressors that do not natively support
    multi-target regression.

    You can use either single estimator, either list of estimators.

    Parameters
    ----------
    estimator : estimator object, or list of estimators, shape (n_targets, )
        An estimator object implementing <fit> and <predict>.
        Or a list of estimators.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for <fit>. None means 1.
        ``-1`` means using all processors.

        When individual estimators are fast to train or predict
        using `n_jobs>1` can result in slower performance due
        to the overhead of spawning processes.

    """

    def __init__(self, estimator, scoring=None, weights=None, n_jobs=None):
        self.estimator = estimator
        self.scoring = scoring
        self.weights = weights
        self.n_jobs = n_jobs


    def fit(self, X, Y, sample_weight=None):

        self.targets_ = list(Y.columns)
        self.classes_ = [LabelBinarizer().fit(y).classes_ for _, y in Y.items()]

        self.estimators_ = check_estimator(self.estimator, self.targets_, 'classifier')

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(clone(e), X, Y[target])
            for e, target in zip(self.estimators_, self.targets_))

        return self

    def score(self, X, Y, *args, **kwargs):
        scores = []
        for estimator, target in zip(self.estimators_, self.targets_):
            scorer = check_scoring(estimator, self.scoring)
            score = scorer(estimator, X, Y[target], *args, **kwargs)
            scores.append(score)
        return np.average(scores, weights=self.weights)

    @property
    def feature_importances_(self):
        imps = [e.feature_importances_ for e in self.estimators_]
        return np.concatenate(imps).mean(axis=0)

    @property
    def coef_(self):
        imps = [e.coef_ for e in self.estimators_]
        return np.concatenate(imps).mean(axis=0)

    def predict(self, X):
        return _call_estimator(self, X, 'predict')

    def predict_proba(self, X):
        return _call_estimator(self, X, 'predict_proba')

    def decision_function(self, X):
        return _call_estimator(self, X, 'decision_function')




def _fit_estimator(estimator, X, y, sample_weight=None):
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator



def _call_estimator(estimator, X, method):

    check_is_fitted(estimator, 'estimators_')

    call_estimator = lambda e: getattr(e, method)(X)

    Y = Parallel(n_jobs=estimator.n_jobs)(delayed(call_estimator)(e)
        for e in estimator.estimators_)

    return Y



def check_estimator(estimator, targets, estimator_type='regressor'):

    if getattr(estimator, '_estimator_type', None) is estimator_type:
        estimators_list = [clone(estimator) for target in targets]

    elif isinstance(estimator, Iterable):
        estimators_list = estimator

        n_est = len(estimators_list)
        n_tar = len(targets)

        if n_est != n_tar:
            raise ValueError("If passed list of estimators, number of "
                             "estimators \n\t\tshould be equal to Y.shape[1]. "
                             "\n\t\tFound: n_estimators = {}, n_targets = {} "
                             " ".format(n_est, n_tar))

        for i, estimator in enumerate(estimators_list):
            if getattr(estimator, '_estimator_type', None) is not estimator_type:
                raise ValueError("If passed list of estimators, each "
                                 "estimator should be {}.\n"
                                 "Error with index {}.".format(estimator_type, i))

    else:
        raise TypeError("Unknown type of <estimator> passed.\n"
                        "Should be {} or list of {}s."
                        " ".format(estimator_type, estimator_type))

    return estimators_list
