from sklearn.calibration import CalibratedClassifierCV

import numpy as np


__all__ = ['CalibratedClassifierCV']




class CalibratedClassifierCV(CalibratedClassifierCV):

    @property
    def feature_importances_(self):
        attr = 'feature_importances_'
        imps = [getattr(clf.base_estimator, attr) for clf in self.calibrated_classifiers_]
        return np.mean(imps, axis=0)

    @property
    def coef_(self):
        attr = 'coef_'
        coef = [getattr(clf.base_estimator, attr) for clf in self.calibrated_classifiers_]
        return np.mean(coef, axis=0)
