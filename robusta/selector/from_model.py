from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import check_cv

from . import _Selector
from robusta.importance import get_importance


class SelectFromModel(_Selector):
    """
    Meta-transformer for selecting features based on importance weights.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        This can be both a fitted (if cv='prefit') or a non-fitted estimator.
        The estimator must have either a <feature_importances_> or <coef_>
        attribute after fitting.

    threshold : string, float, optional (default None)
        The threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded. If "median" (resp. "mean"), then the <threshold> value is
        the median (resp. the mean) of the feature importances. A scaling
        factor (e.g., "1.25*mean") may also be used. If None, drop features
        only based on <max_features>.

    max_features : int, float or None, optional (default 0.5)
        The maximum number of features selected scoring above <threshold>.
        If float, interpreted as proportion of all features.

        To disable <threshold> and only select based on <max_features>,
        set <threshold> to -np.inf.

    cv : int, cross-validation generator, iterable or "prefit"
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

            - None, to disable cross-validation and train single estimator
            on whole dataset (default).
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.
            - "prefit" string constant.

        If "prefit" is passed, it is assumed that <estimator> has been
        fitted already and <fit> function will raise error.

    Attributes
    ----------
    estimator_ : list of fitted estimators, or single fitted estimator
        If <cv> is 'prefit'. If <cv> is None, return single estimator.
        Otherwise return list of fitted estimators, length (n_folds, ).

    feature_importances_ : Series of shape (n_features, )
        Feature importances, extracted from estimator(s)

    threshold_ : float
        The threshold value used for feature selection

    max_features_ : int
        Maximum number of features for feature selection

    use_cols_ : list of str
        Columns to select

    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cv: Optional[int] = None,
        threshold: Optional[float] = None,
        max_features: Optional[float] = None,
    ):
        self.estimator = estimator
        self.threshold = threshold
        self.max_features = max_features
        self.cv = cv

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None
    ) -> "SelectFromModel":
        """
        Fit the estimator to the data and select the most important features.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input data.
        y : pandas Series of shape (n_samples,)
            The target values.
        groups : pandas Series, optional (default=None)
            The group labels for each sample. Used when cross-validating the
            estimator.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        # If 'cv=prefit', estimator should have already been fit
        if self.cv is "prefit":
            raise NotFittedError("Since 'cv=prefit', call transform directly")

        # If no cross-validation is performed, fit the estimator to the full dataset
        elif self.cv is None:
            self.estimator_ = clone(self.estimator).fit(X, y)

        # Otherwise, perform cross-validation
        else:
            self.estimator_ = []

            # Check cross-validation strategy and set up CV splitter
            cv = check_cv(self.cv, y, is_classifier(self.estimator_))

            for trn, _ in cv.split(X, y, groups):
                X_trn, y_trn = X.iloc[trn], y.iloc[trn]

                # Clone and fit estimator on training data
                estimator = clone(self.estimator).fit(X_trn, y_trn)
                self.estimator_.append(estimator)

        return self

    @property
    def feature_importances_(self):
        """
        The importance scores of each feature, averaged over cross-validation folds.

        Returns
        -------
        pandas Series
            The importance scores of each feature.
        """
        imps = []

        # If 'cv=prefit', there's only one estimator to consider
        if self.cv is "prefit":
            estimators = [self.estimator]

        # If no cross-validation is performed, there's only one estimator to consider
        elif self.cv is None:
            estimators = [self.estimator_]

        # Otherwise, there are multiple estimators to consider
        else:
            estimators = self.estimator_

        # Compute feature importances for each estimator and concatenate
        for estimator in estimators:
            imp = get_importance(estimator)
            imps.append(imp)

        return pd.concat(imps, axis=1).mean(axis=1)

    def get_features(self) -> list:
        """
        Return the list of selected features based on the importance scores.

        Returns
        -------
        list
            The list of selected feature names.
        """

        # Compute feature importances
        imp = self.feature_importances_

        # Check threshold and max features and create masks for features that pass
        self.threshold_ = _check_threshold(imp, self.threshold)
        threshold_mask = imp >= self.threshold_

        self.max_features_ = _check_max_features(imp, self.max_features)
        ranking_mask = imp.rank(ascending=False) <= self.max_features_

        # Return list of selected features
        use_cols = imp.index[threshold_mask & ranking_mask]
        return list(use_cols)


def _check_max_features(
    importances: pd.Series, max_features: Union[int, float, None]
) -> int:
    """ "
    Determine the maximum number of features to select.

    Parameters
    ----------
    importances : pd.Series
        The feature importances from which the maximum number of features should be determined.
    max_features : int, float, optional
            The maximum number of features to select. If a float between 0 and 1, it will be interpreted as
        the fraction of the total number of features to select.

    Returns
    -------
    num : int
        The maximum number of features to select.
    """

    # Determine the number of features in the data
    n_features = len(importances)

    # If max_features is not specified, select all features
    if max_features is None:
        max_features = n_features

    # If max_features is an integer, set it as the maximum number of features to use
    elif isinstance(max_features, int):
        max_features = min(n_features, max_features)

    # If max_features is a float, set it as the percentage of total features to use
    elif isinstance(max_features, float):
        max_features = int(n_features * max_features)

    return max_features


def _check_threshold(
    importances: pd.Series, threshold: Union[str, float, None]
) -> float:
    """
    Interpret the threshold value.

    Parameters
    ----------
    importances : ndarray
        Array of feature importances.
    threshold : tr, float, optional
        The threshold value.

    Returns
    -------
    float:
        The interpreted threshold value.

    Raises
    ------
    ValueError:
        If threshold argument is not recognized.
    """

    # If threshold is None, set it to -inf
    if threshold is None:
        threshold = -np.inf

    # If threshold is a string
    elif isinstance(threshold, str):
        # Check if it contains a scaling factor and a reference value
        if "*" in threshold:
            scale, reference = threshold.split("*")
            scale = float(scale.strip())
            reference = reference.strip()

            # Check if the reference value is "median"
            if reference == "median":
                reference = np.median(importances)

            # Check if the reference value is "mean"
            elif reference == "mean":
                reference = np.mean(importances)
            else:
                raise ValueError("Unknown reference: " + reference)

            # Set the threshold to the scaled reference value
            threshold = scale * reference

        # If the threshold is "median", set it to the median of the importances
        elif threshold == "median":
            threshold = np.median(importances)

        # If the threshold is "mean", set it to the mean of the importances
        elif threshold == "mean":
            threshold = np.mean(importances)

        # If the threshold is not recognized, raise a ValueError
        else:
            raise ValueError(
                "Expected threshold='mean' or threshold='median' " "got %s" % threshold
            )

    # If the threshold is a float, leave it as is
    else:
        threshold = float(threshold)

    return threshold
