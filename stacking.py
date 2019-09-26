import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from robusta.crossval._predict import _predict, _check_avg, _avg_preds
from robusta.crossval.base import crossval

from sklearn.model_selection import check_cv
from sklearn.base import (
    clone,
    is_regressor,
    is_classifier,
    BaseEstimator,
    TransformerMixin,
    RegressorMixin,
    ClassifierMixin,
)


__all__ = [
    'StackingTransformer',
    'StackingClassifier',
    'StackingRegressor',
]



class StackingTransformer(BaseEstimator, TransformerMixin):
    '''Stacking Transformer with inbuilt Cross-Validation

    Parameters
    ----------
    estimators : list
        List of (name, estimator) tuples (implementing fit/predict).

    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy:

        - None, use non-CV predictions
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An iterable yielding (train, test) splits as arrays of indices.

    scoring : string, callable or None, optional, default: None
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only a single value.
        If None, the estimator's default scorer (if available) is used.

    test_avg : bool (default=True)
        Stacking strategy (essential parameter).

        See robusta.crossval for details.

    avg_type : string, {'soft', 'hard', 'auto', 'rank'} (default='auto')
        Averaging strategy for aggregating different CV folds predictions.

        See 'crossval' from 'robusta.crossval' for details.

    method : {'predict', 'predict_proba'}, optional (defaul='predict')
        Invokes the passed method name of the passed estimators.

    join_X : bool (default=False)
        If True, concatenate stacked predictions with the original data

    n_jobs : int or None, optional (default=-1)
        Number of jobs to run in parallel. None means 1.

    verbose : int (default=1)
        Verbosity level

    n_digits : int (default=4)
        Verbose score precision

    '''
    def __init__(self, estimators, cv=5, scoring=None, test_avg=True,
                 avg_type='auto', method='predict', join_X=False, n_jobs=-1,
                 verbose=0, n_digits=4, random_state=0):

        self.estimators = estimators
        self.cv = cv
        self.scoring = scoring

        self.test_avg = test_avg
        self.avg_type = avg_type
        self.method = method
        self.join_X = join_X

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_digits = n_digits
        self.random_state = random_state


    def transform(self, X):

        if self._is_train(X):
            S = self._transform_train(X)
        else:
            S = self._transform(X)

        if self.join_X:
            return X.join(S)
        else:
            return S


    def fit(self, X, y, groups=None):

        _check_estimator_types(self.estimators)
        _check_estimator_names(self.estimators)

        self._save_train(X, y)
        self._fit_1st_layer(X, y, groups)

        return self


    def _fit_1st_layer(self, X, y, groups):

        self.names_ = [name for name, _ in self.estimators]

        self.estimators_A_ = []
        self.estimators_B_ = []
        self.scores_std_ = []
        self.scores_ = []

        cv = check_cv(self.cv, y, is_classifier(self.estimators[0]))
        self.folds_ = list(cv.split(X, y, groups))

        for name, estimator in self.estimators:

            result = crossval(estimator, self.folds_, X, y, groups, X_new=None,
                              test_avg=self.test_avg, avg_type=self.avg_type,
                              scoring=self.scoring, method=self.method,
                              verbose=self.verbose, n_digits=self.n_digits,
                              random_state=self.random_state, n_jobs=self.n_jobs,
                              return_estimator=True)

            estimators = result['estimator']
            if self.test_avg:
                self.estimators_A_.append(estimators)
                self.estimators_B_ = None
            else:
                self.estimators_A_.append(estimators[:-1])
                self.estimators_B_.append(estimators[-1:])

            scores = result['score']
            self.scores_.append(np.mean(scores))
            self.scores_std_.append(np.std(scores))

        return self


    def _save_train(self, X, y):
        self._train_shape = X.shape
        self._train_index = X.index
        self._y = y.copy()


    def _is_train(self, X):
        if (X.shape is self._train_shape) and (X.index is self._train_index):
            return True
        else:
            return False


    def _transform_train(self, X):

        pred_list = []
        for estimators in self.estimators_A_:

            avg, method = _check_avg(estimators[0], self.avg_type, self.method)

            preds = Parallel(n_jobs=self.n_jobs)(
                (delayed(_predict)(estimator, method, X.iloc[oof], self._y)
                for estimator, (trn, oof) in zip(estimators, self.folds_)))

            pred = _avg_preds(preds, avg, X, self._y)
            pred_list.append(pred)

        S = _stack_preds(pred_list, self.names_)
        return S


    def _transform(self, X):

        if self.test_avg:
            estimators_list = self.estimators_A_
        else:
            estimators_list = self.estimators_B_

        pred_list = []
        for estimators in estimators_list:

            avg, method = _check_avg(estimators[0], self.avg_type, self.method)

            preds = Parallel(n_jobs=self.n_jobs)(
                (delayed(_predict)(estimator, method, X, self._y)
                for estimator in estimators))

            pred = _avg_preds(preds, avg, X, self._y)
            pred_list.append(pred)

        S = _stack_preds(pred_list, self.names_)
        return S




class StackingRegressor(StackingTransformer, RegressorMixin):
    '''Stacking Regressor with inbuilt Cross-Validation

    Parameters
    ----------
    estimators : list
        List of (name, estimator) tuples (implementing fit/predict).

    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy:

        - None, use non-CV predictions
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An iterable yielding (train, test) splits as arrays of indices.

    scoring : string, callable or None, optional, default: None
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only a single value.
        If None, the estimator's default scorer (if available) is used.

    test_avg : bool (default=True)
        Stacking strategy (essential parameter).

        See 'crossval' from 'robusta.crossval' for details.

    join_X : bool (default=False)
        If True, concatenate stacked predictions with the original data

    n_jobs : int or None, optional (default=-1)
        Number of jobs to run in parallel. None means 1.

    verbose : int (default=1)
        Verbosity level

    n_digits : int (default=4)
        Verbose score precision

    '''
    def __init__(self, estimators, meta_estimator, cv=5, scoring=None,
                 test_avg=True, join_X=False, n_jobs=-1, verbose=0, n_digits=4,
                 random_state=0):

        self.estimators = estimators
        self.meta_estimator = meta_estimator
        self.cv = cv
        self.scoring = scoring

        self.test_avg = test_avg
        self.join_X = join_X

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_digits = n_digits
        self.random_state = random_state


    def fit(self, X, y, groups=None):

        _check_estimator_types(self.estimators, 'regressor')
        _check_estimator_names(self.estimators)

        self._save_train(X, y)
        self._fit_1st_layer(X, y, groups)
        self._fit_2nd_layer(X, y, groups)

        return self


    def _fit_2nd_layer(self, X, y, groups):

        S = self.transform(X)
        self.meta_estimator_ = clone(self.meta_estimator).fit(S, y)


    def predict(self, X):

        S = self.transform(X)
        y = self.meta_estimator_.predict(S)

        return y




class StackingClassifier(StackingTransformer, ClassifierMixin):
    '''Stacking Transformer with inbuilt Cross-Validation

    Parameters
    ----------
    estimators : list
        List of (name, estimator) tuples (implementing fit/predict).

    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy:

        - None, use non-CV predictions
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An iterable yielding (train, test) splits as arrays of indices.

    scoring : string, callable or None, optional, default: None
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only a single value.
        If None, the estimator's default scorer (if available) is used.

    test_avg : bool (default=True)
        Stacking strategy (essential parameter).

        See robusta.crossval for details.

    avg_type : string, {'soft', 'hard', 'auto', 'rank'} (default='auto')
        Averaging strategy for aggregating different CV folds predictions.

        See 'crossval' from 'robusta.crossval' for details.

    method : {'predict', 'predict_proba'}, optional (defaul='predict')
        Invokes the passed method name of the passed estimators.

    join_X : bool (default=False)
        If True, concatenate stacked predictions with the original data

    n_jobs : int or None, optional (default=-1)
        Number of jobs to run in parallel. None means 1.

    verbose : int (default=1)
        Verbosity level

    n_digits : int (default=4)
        Verbose score precision

    '''
    def __init__(self, estimators, meta_estimator, cv=5, scoring=None,
                 test_avg=True, avg_type='auto', method='predict', join_X=False,
                 n_jobs=-1, verbose=0, n_digits=4, random_state=0):

        self.estimators = estimators
        self.meta_estimator = meta_estimator
        self.cv = cv
        self.scoring = scoring

        self.test_avg = test_avg
        self.avg_type = avg_type
        self.method = method
        self.join_X = join_X

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_digits = n_digits
        self.random_state = random_state


    def fit(self, X, y, groups=None):

        _check_estimator_types(self.estimators, 'classifier')
        _check_estimator_names(self.estimators)

        self._save_train(X, y)
        self._fit_1st_layer(X, y, groups)
        self._fit_2nd_layer(X, y, groups)

        return self


    def _fit_2nd_layer(self, X, y, groups):

        S = self.transform(X)
        self.meta_estimator_ = clone(self.meta_estimator).fit(S, y)


    def predict(self, X):

        S = self.transform(X)
        y = self.meta_estimator_.predict(S)

        return y


    def predict_proba(self, X):

        S = self.transform(X)
        y = self.meta_estimator_.predict_proba(S)

        return y


    @property
    def classes_(self):
        return self.meta_estimator_.classes_



def _stack_preds(pred_list, names):

    for name, pred in zip(names, pred_list):
        if hasattr(pred, 'columns'):
            cols = ['{}__{}'.format(name, col) for col in pred.columns]
            pred.columns = cols
        else:
            pred.name = name

    pred = pd.concat(pred_list, axis=1)
    return pred



def _check_estimator_types(estimators, allow_types=['classifier', 'regressor']):

    est_types = np.array([e._estimator_type for _, e in estimators])
    allow_types = np.array(allow_types)

    if not (est_types == est_types[0]).all():
        raise ValueError('Estimator types must be the same')

    if not (allow_types == est_types[0]).any():
        raise ValueError('Estimator types must be in: {}'.format(allow_types))



def _check_estimator_names(estimators):

    names = np.array([name for name, _ in estimators])
    unames = np.unique(names)

    if unames.shape != names.shape:
        raise ValueError('Estimator names must be unique')
