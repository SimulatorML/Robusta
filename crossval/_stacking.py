import pandas as pd
import numpy as np

from ._crossval import cross_val, cross_val_pred, _extract_est_name


#__all__ = ['stacking', 'StackingTransformer']
__all__ = ['stacking']



def stacking(estimators, cv, X, y, groups=None, X_new=None, test_avg=True,
             voting='auto', method='predict', n_jobs=-1, verbose=0):
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
    est_names = []

    # Fit & predict
    for estimator in estimators:

        oof_pred, new_pred = cross_val_pred(estimator, cv, X, y, groups,
            X_new, test_avg, voting, method, n_jobs, verbose)

        oof_preds.append(oof_pred)
        new_preds.append(new_pred)

        name = _extract_est_name(estimator, drop_type=True)
        est_names.append(name)

    # Concat predictions
    oof_stack = pd.concat(oof_preds, axis=1)
    new_stack = pd.concat(new_preds, axis=1)

    # Columns renaming
    oof_stack.columns = est_names
    new_stack.columns = est_names

    return oof_stack, new_stack
