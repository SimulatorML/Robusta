import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from sklearn.base import clone, is_classifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection._split import check_cv
from sklearn.utils.metaestimators import _safe_split

from ..crossval._crossval import _pred, _extract_est_name, _check_voting, _concat_preds
from ..crossval import cross_val, cross_val_pred


__all__ = ['stack', 'Stacker', 'make_stacker']



def stack(estimators, cv, X, y, groups=None, X_new=None, test_avg=True,
          voting='auto', method='predict', join_X=False,
          n_jobs=-1, verbose=0):
    """Get Out-of-Fold and Test predictions of multiple estimators.

    Parameters
    ----------
    estimators : list of estimator objects
        The objects to use to fit the data.

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
    oof_preds : DataFrame, shape [n_samples, n_estimators]
        Out-of-fold predictions

    new_preds : DataFrame, shape [m_samples, n_estimators] or None
        Test predictions (unseen data)
        None if X_new is not defined

    """
    oof_preds = []
    new_preds = []

    # Extract estimator names
    est_names = []

    for estimator in estimators:
        est_names.append(_extract_est_name(estimator, drop_type=True))

    # Fit & predict
    for estimator in estimators:

        oof_pred, new_pred = cross_val_pred(estimator, cv=cv, X=X, y=y,
            groups=groups, X_new=X_new, test_avg=test_avg, voting=voting,
            method=method, n_jobs=n_jobs, verbose=verbose)

        oof_preds.append(oof_pred)
        new_preds.append(new_pred)

    # Concatenate predictions
    S_train = _stack_preds(oof_preds, est_names, join_X, X)
    S_test = _stack_preds(new_preds, est_names, join_X, X_new)

    return S_train, S_test



class Stacker(BaseEstimator, TransformerMixin):
    """Stacker. Scikit-learn compatible API for stacking.

    Parameters
    ----------
    estimators : list of tuples, default None
        Base level estimators.
        If None then by default:
            DummyRegressor (predicts constant 5.5) - for regression task
            DummyClassifier (predicts constant 1) - for classification task
        You can use any sklearn-like estimators.
        Each tuple in the list contains arbitrary
            unique name and estimator object, e.g.:
        estimators = [('lr', LinearRegression()),
                      ('ridge', Ridge(random_state=0))]
        Note. According to sklearn convention for binary classification
            task with probabilities estimator must return probabilities
            for each class (i.e. two columns).

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

    verbose : int, default 0
        Level of verbosity.
        0 - show no messages
        1 - for each estimator show mean score
        2 - for each estimator show score for each fold and mean score

    Attributes
    ----------
    estimators_ : list
        List of base estimators (not fitted) passed by user (or default)

    """
    def __init__(self, estimators, cv=5, scoring=None, test_avg=True,
                 voting='auto', method='predict', join_X=False,
                 n_jobs=-1, verbose=0):

        self.estimators = estimators
        self.scoring = scoring
        self.cv = cv

        self.join_X = join_X
        self.test_avg = test_avg
        self.voting = voting
        self.method = method
        #self.return_importance = return_importance

        self.n_jobs = n_jobs
        self.verbose = verbose


    def fit_transform(self, X, y, groups=None, **fit_params):

        # Get footprint & init attributes
        self._pre_fit(X, y, groups)

        # Fit & Predict for each Estimator
        oof_preds = []

        for estimator in self.estimators_:

            # Fit each estimator
            results = cross_val(estimator, cv=self.folds, X=X, y=y, groups=groups,
                scoring=self.scoring, voting=self.voting, method=self.method,
                X_new=None, test_avg=self.test_avg, return_estimator=True,
                return_pred=True, return_score=True, n_jobs=self.n_jobs)

            # Save predictions
            oof_pred = results['oof_pred']
            oof_preds.append(oof_pred)

            self._save_results(results)

        # Concatenate predicitons
        S_train = _stack_preds(oof_preds, self.est_names_, self.join_X, X)

        return S_train



    def fit(self, X, y, groups=None, **fit_params):

        # Get footprint & init attributes
        self._pre_fit(X, y, groups)

        # Fit each Estimator
        for estimator in self.estimators_:

            # Fit each estimator
            results = cross_val(estimator, cv=self.folds, X=X, y=y, groups=groups,
                scoring=self.scoring, voting=self.voting, method=self.method,
                X_new=None, test_avg=self.test_avg, return_estimator=True,
                return_pred=False, return_score=True, n_jobs=self.n_jobs)

            self._save_results(results)

        return self


    def transform(self, X):

        # Check if passed set is the same as passed in fit/fit_transform
        is_train_set = self._check_train_set(X)

        # X & X_new are processing differently
        if is_train_set:
            S_train = self._transform(X)
            return S_train
        else:
            S_test = self._transform_new(X)
            return S_test



    def _pre_fit(self, X, y, groups=None):

        # Memorize data
        self._save_train_set(X)
        self.target = y.name
        self.encoders = []

        # Init attributes
        self.n_estimators_ = len(self.estimators)
        self.estimators_ = [estimator for _, estimator in self.estimators]
        self.est_names_ = [name for name, _ in self.estimators]

        self.estimators_A_ = [] # estimators for transforming train set (X, y)
        self.estimators_B_ = [] # estimators for transforming test set (X_new)
        self.scores_ = []

        # Check Validation Scheme
        # TODO: check if folds, generated by any estimator, are the same
        # Make shure, that all estimators have the same type
        self.cv_ = check_cv(self.cv, y, classifier=is_classifier(self.estimators[0]))
        self.folds = list(self.cv_.split(X, y, groups))



    def _save_results(self, results):

        # Save Score
        score = results['score']
        self.scores_.append(score)

        # Save Encoder
        encoder = results['encoder']
        self.encoders.append(encoder)

        # Save Fitted Estimators
        estimators = results['estimator']

        if self.test_avg:
            # Stacking Type A
            self.estimators_A_.append(estimators)
            self.estimators_B_ = None

        else:
            # Stacking Type B
            self.estimators_A_.append(estimators[:-1])
            self.estimators_B_.append(estimators[-1])



    def _transform(self, X):

        parallel = Parallel(max_nbytes='256M', pre_dispatch='2*n_jobs',
                            n_jobs=self.n_jobs)

        oof_preds = []
        target = self.target

        for k, estimators in enumerate(self.estimators_A_):

            # Check voting & method for new list of estimators
            method, voting = self.method, self.voting
            method, avg = _check_voting(estimators[0], voting, method)

            # Compute predictions by each estimator
            def _pred_oof(estimator, X, oof):
                X_oof, _ = _safe_split(estimator, X, None, oof)
                return _pred(estimator, method, X_oof, target)

            preds = parallel((delayed(_pred_oof)(estimator, X, fold[1])
                for fold, estimator in zip(self.folds, estimators)))

            # Save averaged predictions
            encoder = self.encoders[k]
            oof_pred = _concat_preds(preds, avg, encoder, target, X.index)
            oof_preds.append(oof_pred)

        # Concatenate predicitons
        S_train = _stack_preds(oof_preds, self.est_names_, self.join_X, X)

        return S_train



    def _transform_new(self, X):

        parallel = Parallel(max_nbytes='256M', pre_dispatch='2*n_jobs',
                            n_jobs=self.n_jobs)

        new_preds = []
        target = self.target

        if self.test_avg:

            # Stacking Type A
            for k, estimators in enumerate(self.estimators_A_):

                # Check voting & method for new list of estimators
                method, voting = self.method, self.voting
                method, avg = _check_voting(estimators[0], voting, method)

                # Compute predictions by each estimator
                preds = parallel((delayed(_pred)(estimator, method, X, target)
                    for estimator in estimators))

                # Save averaged predictions
                encoder = self.encoders[k]
                new_pred = _concat_preds(preds, avg, encoder, target, X.index)
                new_preds.append(new_pred)

        else:

            # Stacking Type B
            for k, estimator in enumerate(self.estimators_B_):

                # Check voting & method for new estimator
                method, voting = self.method, self.voting
                method, avg = _check_voting(estimator, voting, method)

                # Compute prediction
                pred = _pred(estimator, method, X, target)

                # Save predictions
                encoder = self.encoders[k]
                new_pred = _concat_preds([pred], avg, encoder, target, X.index)
                new_preds.append(new_pred)

        # Concatenate predicitons
        S_test = _stack_preds(new_preds, self.est_names_, self.join_X, X)

        return S_test



    def _save_train_set(self, X, **kwargs):
        self.train_shape = X.shape
        self.train_footprint = _get_footprint(X, **kwargs)



    def _check_train_set(self, X, **kwargs):
        if X.shape == self.train_shape:
            return _check_footprint(X, self.train_footprint, **kwargs)
        else:
            return False



def make_stacker(estimators, cv, X, y, groups=None, X_new=None, test_avg=True,
                 voting='auto', method='predict', join_X=False,
                 n_jobs=-1, verbose=0):

    # Extract estimator names
    est_names = []

    for estimator in estimators:
        est_names.append(_extract_est_name(estimator, drop_type=True))

    estimators = list(zip(est_names, estimators))

    # Init Stacker
    stacker = Stacker(estimators, cv, X, y, groups=groups, X_new=X_new,
                      test_avg=test_avg, voting=voting, method=method,
                      join_X=join_X, n_jobs=n_jobs, verbose=verbose)

    return stacker




def _get_footprint(X, n_items=1000):
    footprint = []
    n_rows, n_cols = X.shape
    n = n_rows * n_cols
    # np.random.seed(0) # for development
    inds = np.random.choice(n, min(n_items, n), replace=False)

    for ind in inds:
        i = ind // n_cols
        j = ind - i * n_cols
        val = X.iloc[i, j]
        footprint.append((i, j, val))

    return footprint



def _check_footprint(X, footprint, rtol=1e-05, atol=1e-08, equal_nan=False):
    try:
        for i, j, val in footprint:
            assert np.isclose(X.iloc[i, j], val, rtol=rtol, atol=atol, equal_nan=equal_nan)
        return True
    except AssertionError:
        return False



def _stack_preds(preds, names, join_X=False, X=None):

    S = pd.concat(preds, axis=1)
    S.columns = names
    # FIXME: crashes on non-binary classification
    # Use MultiIndex columns in this cases

    if join_X:
        return X.join(S)
    else:
        return S
