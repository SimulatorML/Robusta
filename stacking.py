import pandas as pd
import numpy as np

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
    def __init__(self, estimators, cv=5, scoring=None, test_avg=True, avg_type='auto',
                 method='predict', join_X=False, n_jobs=-1, verbose=0, n_digits=4,
                 random_state=0):

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

        return X.join(S) if join_X else S


    def fit(self, X, y, groups=None):

        self._save_train(X)
        self._fit_1st_layer(X, y, groups)

        return self


    def _fit_1st_layer(self, X, y, groups):

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


    def _save_train(self, X):
        self._train_shape = X.shape
        self._train_index = X.index


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
                (delayed(_predict)(estimator, method, X.iloc[oof], y)
                for estimator, (_, oof) in zip(estimators, self.folds_)))

            pred = _avg_preds(preds, avg, X, y)
            pred_list.append(pred)

        S = pd.concat(pred_list, axis=1) # TODO: estimators' names
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
                (delayed(_predict)(estimator, method, X, y)
                for estimator in estimators))

            pred = _avg_preds(preds, avg, X, y)
            pred_list.append(pred)

        S = pd.concat(pred_list, axis=1) # TODO: estimators' names
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
    def __init__(self, estimators, meta_estimator, cv=5, scoring=None, test_avg=True,
                 join_X=False, n_jobs=-1, verbose=0, n_digits=4, random_state=0):

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

        self._save_train(X)
        self._fit_1st_layer(X, y, groups)
        self._fit_2nd_layer(X, y, groups)

        return self


    def _fit_2nd_layer(self, X, y):

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

        self._save_train(X)
        self._fit_1st_layer(X, y, groups)
        self._fit_2nd_layer(X, y, groups)

        return self


    def _fit_2nd_layer(self, X, y):

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
