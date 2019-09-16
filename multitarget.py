import numpy as np
import pandas as pd

from collections.abc import Iterable

from sklearn.base import is_regressor
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.fixes import parallel_helper

__all__ = [
    'MultiTargetRegressor',
    #'MultiTargetClassifier',
]



def _fit_estimator(estimator, X, y, sample_weight=None):
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator



class MultiTargetRegressor(BaseEstimator, RegressorMixin):
    """Multi target regression

    This strategy consists of fitting one regressor per target. This is a
    simple strategy for extending regressors that do not natively support
    multi-target regression.

    You can use either single estimator, either list of estimators.

    Parameters
    ----------
    estimator : estimator object, or list of estimators, shape (n_outputs, )
        An estimator object implementing <fit> and <predict>.
        Or a list of estimators.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for <fit>. None means 1.
        ``-1`` means using all processors.

        When individual estimators are fast to train or predict
        using `n_jobs>1` can result in slower performance due
        to the overhead of spawning processes.

    """

    def __init__(self, estimator, n_jobs=None):
        self.estimator = estimator
        self.n_jobs = n_jobs


    def fit(self, X, Y, sample_weight=None):
        """Fit the model to data.

        Fit a separate model for each output variable.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
        Y : DataFrame, shape (n_samples, n_outputs)

        sample_weight : array-like, shape = (n_samples) or None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        Returns
        -------
        self : object

        """
        self.targets_ = list(Y.columns)

        if is_regressor(self.estimator):
            self.estimators_ = [clone(self.estimator) for target in self.targets_]

        elif isinstance(self.estimator, Iterable):
            self.estimators_ = self.estimator

            if len(estimators) != len(self.targets_):
                raise ValueError("If passed list of estimators, number of "
                                 "estimators should be equal to Y.shape[1]")

            for i, estimator in enumerate(self.estimators_):
                if not is_regressor(estimator):
                    raise ValueError("If passed list of estimators, each "
                                     "estimator should be regressor.\n"
                                     "Error with index {}.".format(i))

        else:
            raise TypeError("Unknown type of <estimator> passed.\n"
                            "Should be regressor or list of regressors.")

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(clone(e), X, Y[target])
            for e, target in zip(self.estimators_, self.targets_))

        return self


    def predict(self, X):
        """Predict multi-output variable using a model
         trained for each target variable.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Data.

        Returns
        -------
        Y : DataFrame, shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.

        """
        check_is_fitted(self, 'estimators_')

        Y = Parallel(n_jobs=self.n_jobs)(delayed(parallel_helper)(e, 'predict', X)
                for e in self.estimators_)

        Y = pd.concat([pd.Series(y) for y in Y], axis=1)
        Y.columns = self.targets_
        Y.index = X.index

        return Y
