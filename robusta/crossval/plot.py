from collections import defaultdict
from typing import Union, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy import interp, stats
from sklearn.metrics import roc_curve

from . import _cat_staged_predict, _lgb_staged_predict, _xgb_staged_predict
from .result import check_cvs


def plot_learning_curve(result: dict,
                        X: pd.DataFrame,
                        y: pd.Series,
                        groups: Optional[pd.Series] = None,
                        max_iter: int = 0,
                        step: int = 1,
                        mode: str = 'mean',
                        train_score: bool = False,
                        n_jobs: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot the learning curve of a model using cross-validation.

    Args:
        result (dict): The result dictionary obtained from a call to a
            `scikit-learn` `GridSearchCV` or `RandomizedSearchCV` object.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        groups (Optional[pd.Series], optional): The groups used for grouping
            samples in the `cv.split` method of the cross-validator. Defaults to None.
        max_iter (int, optional): The maximum number of iterations to plot.
            If 0, the maximum number of iterations is determined by the number
            of iterations of the estimators. Defaults to 0.
        step (int, optional): The step size between iterations. Defaults to 1.
        mode (str, optional): The mode of the plot, either 'mean', 'fold', or 'both'.
            'mean' plots the mean score across all folds, 'fold' plots the score
            of each individual fold, and 'both' plots both. Defaults to 'mean'.
        train_score (bool, optional): Whether to plot the training score in addition
            to the validation score. Defaults to False.
        n_jobs (Optional[int], optional): The number of jobs to run in parallel for
            computing the scores. If -1, use all available CPUs. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays: the training
            scores (shape = (n_folds, n_iterations)) and the validation scores
            (shape = (n_folds, n_iterations)).

    Raises:
        AssertionError: If the specified `mode` is not one of 'mean', 'fold', or 'both'.
        NotImplementedError: If the estimator is not supported.
    """

    # Get the estimators, scorer, and cv from the `result` dictionary
    estimators = result['estimator']
    scorer = result['scorer']
    cv = result['cv']

    # Check that the specified mode is valid
    modes = ('mean', 'fold', 'both')

    # Assert that the `mode` argument is one of the valid modes
    assert mode in modes, f'<mode> must be from {modes}. Found {mode}'

    # Determine the generator function for the estimator and the maximum number of iterations
    estimator = estimators[0]
    name = estimator.__class__.__name__

    # Check which model is used and get the generator and maximum number of iterations accordingly
    if name.startswith('CatBoost'):
        generator = _cat_staged_predict
        if max_iter == 0:
            max_iter = min([e.tree_count_ for e in estimators])
    elif name.startswith('LGB'):
        generator = _lgb_staged_predict
        if max_iter == 0:
            max_iter = min([e.booster_.num_trees() for e in estimators])
    elif name.startswith('XGB'):
        raise NotImplementedError('XGBoost currently does not supported')
        generator = _xgb_staged_predict
        if max_iter == 0:
            max_iter = min([e.n_estimators for e in estimators])
    else:
        raise NotImplementedError('Only LGBM and CatBoost currently supported')

    # Define the predictor object based on whether the estimator is a classifier or regressor
    if estimator._estimator_type == 'classifier':
        predictor = _StagedClassifier()
    elif estimator._estimator_type == 'regressor':
        predictor = _StagedRegressor()

    # Define the stages
    stages = np.arange(step, max_iter + step, step)

    # Split the data into folds using the cv object
    folds = cv.split(X, y, groups)

    # Calculate the training and validation scores for each estimator and fold
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_get_scores)(estimator, generator, predictor, trn, val, X, y,
                             scorer, max_iter, step, train_score)
        for (trn, val), estimator in zip(folds, estimators)
    )

    # Get the training and validation scores as numpy arrays
    trn_scores = np.array([s[0] for s in scores])
    val_scores = np.array([s[1] for s in scores])

    # Create a plot of the learning curve
    plt.figure()
    if not train_score:
        trn_scores = None
    else:
        # Calculate the mean and standard deviation of the training scores
        avg = trn_scores.mean(axis=0)
        std = trn_scores.std(axis=0)

        # Plot the mean training score and the region between the mean minus and plus one standard deviation
        if mode in ['mean', 'both']:
            plt.fill_between(stages, avg - std, avg + std, alpha=.1, color='b')
            plt.plot(stages, avg, label='train score', color='b')

        # Plot the individual training scores for each fold
        if mode in ['fold', 'both']:
            for scores in trn_scores:
                plt.plot(stages, scores, '--', color='b', lw=0.5, alpha=0.5)
    if True:
        # Calculate the mean and standard deviation of the validation scores
        avg = val_scores.mean(axis=0)
        std = val_scores.std(axis=0)
        # Plot the mean validation score and the region between the mean minus and plus one standard
        if mode in ['mean', 'both']:
            plt.fill_between(stages, avg - std, avg + std, alpha=.1, color='y')
            plt.plot(stages, avg, label='valid score', color='y')

        # Plot the individual validation scores for each fold
        if mode in ['fold', 'both']:
            for scores in val_scores:
                plt.plot(stages, scores, '--', color='y', lw=0.5, alpha=0.5)
    plt.legend()
    plt.show()
    return trn_scores, val_scores


def plot_ttest(resultA: dict,
               resultB: dict,
               score: str = 'val_score',
               label: str = 'label',
               cuped: bool = False) -> Tuple[float, float]:
    """
    Plots a t-test comparing the scores in two dictionaries.

    Parameters:
    -----------
    resultA : dict
        Dictionary containing the results of group A.
    resultB : dict
        Dictionary containing the results of group B.
    score : str, optional
        The key of the score values in the dictionaries. Default is 'val_score'.
    label : str, optional
        The key of the group labels in the dictionaries. Default is 'label'.
    cuped : bool, optional
        Whether to apply CUPED (control variates for unbiased estimation of the treatment effect).
        Default is False.

    Returns:
    --------
    Tuple of two floats representing the t-test statistic and p-value.

    """

    # Extract the scores from the dictionaries
    a = resultA[score]
    b = resultB[score]

    # Check that the scores have the same length
    assert len(a) == len(b), 'Both scores must be of the same size'

    # Get the number of samples
    n = len(a)

    # Set the labels for the plot
    labels = ['0', '1']
    if label in resultA: labels[0] = resultA[label]
    if label in resultB: labels[1] = resultB[label]

    # Apply CUPED (control variates for unbiased estimation of the treatment effect) if required
    if cuped:
        theta = np.cov(a, b)[0, 1] / np.var(a)
        b -= (a - np.mean(b)) * theta

    # Calculate the t-test statistic and p-value
    t, p = stats.ttest_rel(a, b)

    # Create a 2x2 plot
    _, axes = plt.subplots(2, 2)

    # Plot boxplots for the two groups
    ax = axes[0, 0]
    sns.boxplot(labels, [a, b], linewidth=2.0, ax=ax)
    ax.grid(alpha=0.2)

    # Plot individual data points and means for the two groups
    ax = axes[1, 0]
    for x, y in zip(a, b):
        ax.plot(labels, [x, y], 'o-', color='b', alpha=0.8)
    ax.plot(labels, [np.mean(a), np.mean(b)], 'o-', color='w')
    ax.grid(alpha=0.2)

    # Plot histograms for the two groups
    ax = axes[0, 1]
    sns.distplot(a, 10, label=labels[0], ax=ax)
    sns.distplot(b, 10, label=labels[1], ax=ax)
    ax.grid(alpha=0.2)
    ax.legend()

    # Plot the t-distribution and shade the areas representing the p-value
    ax = axes[1, 1]
    x_abs = max(5, abs(t))
    x_min, x_max = -x_abs, +x_abs
    xx = np.arange(t, x_max, 0.001)
    yy = stats.t.pdf(xx, n - 1)
    ax.plot(xx, yy, color='gray')
    ax.fill_between(xx, yy, color='gray', alpha=0.2)
    xx = np.arange(x_min, t, 0.001)
    yy = stats.t.pdf(xx, n - 1)
    ax.plot(xx, yy, color='r')
    ax.fill_between(xx, yy, color='r', alpha=0.2)
    ax.legend(['t-value = {:.4f}'.format(t),
               'p-value = {:.4f}'.format(p)])
    ax.grid(alpha=0.2)

    # Return the t-test statistic and p-value
    return t, p


def plot_roc_auc(results: Union[list, dict],
                 X: pd.DataFrame,
                 y: pd.Series,
                 groups: Optional[pd.Series] = None,
                 labels: Optional[list] = None,
                 colors: Optional[list] = None,
                 steps: int = 200) -> None:
    """
    Plots the Receiver Operating Characteristic (ROC) curve with the area under the curve (AUC) for multiple classifiers.

    Parameters:
    results (Union[List, Dict]): A list or dictionary of dictionaries, where each dictionary has an 'estimator' key
        pointing to a trained classifier that has a 'predict_proba' method. If a list, each estimator is assumed
        to use the same cross-validation object, otherwise each dictionary must contain a 'cv' key pointing to a
        cross-validation object.
    X (pd.DataFrame): The input data used for training the classifiers.
    y (pd.Series): The target data used for training the classifiers.
    groups (Optional[pd.Series]): An array containing group labels for the samples, used for group-based cross-validation.
        Default is None.
    labels (Optional[List]): A list of strings to label the ROC curve for each classifier. If None, the label for each
        classifier will be an integer based on its index. Default is None.
    colors (Optional[List]): A list of color strings to use for each ROC curve. If None, a default color cycle is used.
        Default is None.
    steps (int): The number of steps to use when interpolating the ROC curve. Default is 200.

    Returns:
    None. The plot is displayed using Matplotlib.

    Raises:
    AssertionError: If the labels or colors lists are not the same length as the number of classifiers in the results list.
        Also raised if any of the results dictionaries do not contain an 'estimator' key pointing to a classifier object.
    """

    # check cross-validation strategy and generate a cross-validator object
    cv = check_cvs(results=results,
                   X=X,
                   y=y,
                   groups=groups)

    # define error message for label length
    msg = "<labels> must be of same len as <results>"
    if labels:
        # raise error if labels given and not the same length as results
        assert len(labels) == len(results), msg

    else:
        # create labels as list of indices if not given
        labels = list(range(len(results)))

    # define error message for color length
    msg = "<colors> must be of same len as <results>"
    if colors:
        # raise error if colors given and not the same length as results
        assert len(colors) == len(results), msg
    else:
        # use default color cycle if not given
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # define error message for missing estimator
    msg = "Each <result> must have 'estimator' key"
    for result in results:
        # raise error if any result is missing estimator key
        assert 'estimator' in result, msg

    # create array of average false positive rates
    avg_fpr = np.linspace(0, 1, steps)

    # create dictionary to store tpr values for each label
    curves = defaultdict(list)

    # use cross-validation object from first result
    cv = results[0]['cv']

    # loop over cross-validation splits
    for i, (_, oof) in enumerate(cv.split(X, y, groups)):
        # get in-fold training data
        X_oof = X.iloc[oof]
        # get in-fold target values
        y_oof = y.iloc[oof]

        # iterate through results
        for j, result in enumerate(results):
            # get predicted probabilities for in-fold data
            y_pred = result['estimator'][i].predict_proba(X_oof)[:, 1]

            # compute roc curve for in-fold data
            fpr, tpr, _ = roc_curve(y_oof, y_pred)

            # interpolate tpr values to match average fpr values
            tpr = interp(avg_fpr, fpr, tpr)

            # set tpr value at fpr=0 to 0
            tpr[0] = 0.0

            # add interpolated tpr values to dictionary for corresponding label
            curves[labels[j]].append(tpr)

    # use default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # create dictionary of label-color pairs
    colors = dict(zip(labels, colors))

    # create new figure
    plt.figure()

    # iterate through label-tpr pairs
    for label, tprs in curves.items():
        # get color for current label
        c = colors[label]

        # iterate through tpr values for current label
        for tpr in tprs:
            # plot tpr curve with low opacity
            plt.plot(avg_fpr, tpr, c=c, alpha=0.2)

        # get average tpr values across all folds
        avg_tpr = np.mean(tprs, axis=0)

        # plot average tpr curve with label
        plt.plot(avg_fpr, avg_tpr, c=c, label=label)

        # get standard deviation of tpr values across all folds
        std_tpr = np.std(tprs, axis=0)

        # calculate upper bound of tpr values
        tpr_upper = np.minimum(avg_tpr + std_tpr, 1)

        # Calculate the lower bound of the 95% confidence interval for the true positive rate
        tpr_lower = np.maximum(avg_tpr - std_tpr, 0)

        # Fill the area between the upper and lower bounds with the corresponding color and low opacity
        plt.fill_between(avg_fpr, tpr_lower, tpr_upper, color=c, alpha=.1)

    # Add a legend to the plot
    plt.legend(loc='lower right')

    # Show the plot
    plt.show()
