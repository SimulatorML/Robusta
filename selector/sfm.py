import pandas as pd
import numpy as np

from .base import Selector



#SelectFromModel(estimator, threshold=None, prefit=False, norm_order=1, max_features=None)[source]¶

class SelectFromModel(Selector):
    """Meta-transformer for selecting features based on importance weights.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        This can be both a fitted (if cv='prefit') or a non-fitted estimator.
        The estimator must have either a <feature_importances_> or <coef_>
        attribute after fitting.

    threshold : string, float, optional default None
        The threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded. If "median" (resp. "mean"), then the <threshold> value is
        the median (resp. the mean) of the feature importances. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicitly (e.g, Lasso), the threshold used is 1e-5.

        Otherwise, "mean" is used by default.

    max_features : int or None, optional
        The maximum number of features selected scoring above <threshold>.
        To disable <threshold> and only select based on <max_features>,
        set <threshold> to -np.inf.

    cv : int, cross-validation generator, iterable or "prefit"
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

            - None, to disable cross-validation and train single estimator
            on whole dataset.
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.
            - "prefit" string constant (default).

        If "prefit" is passed, it is assumed that <estimator> has been
        fitted already and <fit> function will raise error.

    Attributes
    ----------
    estimator_ : an estimator
        The base estimator from which the transformer is built.
        This is stored only when a non-fitted estimator is passed to the
        <SelectFromModel>, i.e when prefit is False.

    threshold_ : float
        The threshold value used for feature selection.

    """

    def __init__(self, estimator, threshold=None, max_features=None, cv='prefit'):

        self.estimator = estimator
        self.threshold = threshold
        self.max_features = max_features
        self.cv = cv


    def fit(self, X, y, groups=None):

        
