import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator



class ModelCV(BaseEstimator):
    """
    Model with cross-validation wrapper

    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data.

    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the whole dataset for training,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

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

                   Column 'fold' will be added.

        Ignored if <return_pred> set to False, or <method> is not 'predict'.

    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.
        Ignored if return_pred=False.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int (default=1)
        Verbosity level

    n_digits : int (default=4)
        Verbose score(s) precision

    Attributes
    ----------
    result_ : dict
        Array of scores/predictions/time of the estimator for each run of the
        cross validation. If test_avg=True, arrays has shape [n_splits],
        otherwise [n_splits+1] except score & score_time.

        See <crossval> documentation to learn more.

    feature_importances_ : Series of shape (n_feature_)
        Averaged feature importances

    """
    def __init__(self, cv, scoring=None, averaging='auto', method='predict',
                 n_jobs=-1, verbose=1, n_digits=4):

        self.cv = cv
        self.scoring = scoring
        self.averaging = averaging
        self.method = method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_digits = n_digits


    def fit(self, X, y, groups=None):

        self.result_ = crossval(self.estimator, cv=self.cv, X=X, y=y,
                                groups=groups, scoring=self.scoring,
                                averaging=self.averaging, method=self.method,
                                return_estimator=True, return_pred=True,
                                return_score=True, n_jobs=self.n_jobs,
                                verbose=self.verbose, n_digits=self.n_digits)

        try:
            imps = []

            for estimator in self.result_['estimator']:
                imp = extract_importance(estimator)
                imps.append(imp)

            self.feature_importances_ = pd.concat(imps, axis=1).mean(axis=1)

        except:
            pass

        return self


    def predict(self, X):

        if False:
            # TODO: check if X_train
            # TODO: prob or pred?
            return self.result_['oof_pred']

        else:
            # TODO: averaging predictions
            return None


    def score(self, X, y):

        if False:
            # TODO: check if X_train
            # TODO: multimetric case
            return self.result_['score']['score']

        else:
            scorers, is_multimetric = _check_multimetric_scoring(self.estimator,
                                                                 self.scoring)

            for key, scorer in scorers.items():
                # TODO: multimetric case
                if key is 'score':
                    scores = []

                    for estimator in self.result_['estimator']:
                        score = scorer(estimator, X, y)
                        scores.append(score)

                    score = np.mean(scores)
                    return score
