from typing import Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin


class OutlierDetector(BaseEstimator, OutlierMixin):
    """
    OutlierDetector is a custom class that inherits from BaseEstimator and OutlierMixin.
    It can be used to detect and remove outliers from a given dataset.

    Parameters:
    -----------
    None

    Returns:
    --------
    Resampled version of X and y with the outliers removed.
    """

    _estimator_type = "outlier_detector"

    def fit_resample(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fits the model to the input data X and returns a resampled version of X and y with the outliers removed.

        Parameters
        ----------
        X : pd.DataFrame
            The input data of shape (n_samples, n_features).

        y : Optional[pd.Series]
            The target labels of shape (n_samples,) (default=None).

        Returns
        -------
        X_resampled : np.ndarray
            The resampled version of X with the outliers removed.

        y_resampled : Optional[np.ndarray]
            The resampled version of y with the outliers removed (only if y is not None).
        """

        # Use the fit_predict method of the class to get predicted labels for input data X
        labels = self.fit_predict(X, y)

        # Set the boolean mask for outliers as True where predicted labels are less than 0 (outliers)
        out = labels < 0

        # Return the resampled version of input data X and target labels y with the outliers removed
        return X[~out], y[~out]
