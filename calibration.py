import numpy as np
from sklearn.calibration import CalibratedClassifierCV


class CalibratedClassifierCV(CalibratedClassifierCV):
    """
    A subclass of `CalibratedClassifierCV` that computes ensemble feature importances and coefficients.

    This class provides two additional properties:
    - `feature_importances_`: an array of feature importances averaged over the base estimators in the ensemble.
    - `coef_`: an array of coefficients averaged over the base estimators in the ensemble.
    The arrays have the same shape as the corresponding arrays of the base estimator.
    """

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Get the ensemble feature importances.

        Returns
        -------
        np.ndarray:
            An array of feature importances averaged over the base estimators in the ensemble.
            The shape is the same as the corresponding array of the base estimator.
        """

        attr = "feature_importances_"
        imps = [
            getattr(clf.base_estimator, attr) for clf in self.calibrated_classifiers_
        ]

        # Compute the mean of the feature importances over the base estimators.
        return np.mean(imps, axis=0)

    @property
    def coef_(self) -> np.ndarray:
        """
        Get the ensemble coefficients.

        Returns
        -------
        np.ndarray:
            An array of coefficients averaged over the base estimators in the ensemble.
            The shape is the same as the corresponding array of the base estimator.
        """

        attr = "coef_"
        coef = [
            getattr(clf.base_estimator, attr) for clf in self.calibrated_classifiers_
        ]

        # Compute the mean of the coefficients over the base estimators.

        return np.mean(coef, axis=0)
