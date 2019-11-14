import numpy as np
import pandas as pd

from sklearn.base import clone, ClassifierMixin

from sklearn.utils.metaestimators import _BaseComposition, if_delegate_has_method
from robusta.utils import logmsg


__all__ = ['PseudoLabeling',]




class PseudoLabeling(_BaseComposition, ClassifierMixin):

    def __init__(self, estimator, proba=0.95, max_iter=None, verbose=0):
        self.estimator = estimator
        self.max_iter = max_iter
        self.verbose = verbose
        self.proba = proba


    def fit(self, X, y):

        self.X_ = X
        self.y_ = y

        self.estimator_ = clone(self.estimator).fit(X, y)
        self.n_iter_ = 1

        return self


    def partial_fit(self, X, y):

        if hasattr(self, 'estimator_'):
            self.X_ = pd.concat([self.X_, X])
            self.y_ = pd.concat([self.y_, y])

            self.estimator_.fit(self.X_, self.y_)
            self.n_iter_ = self.n_iter_ + 1

        else:
            self.fit(X, y)

        return self


    def predict(self, X):
        return self._pseudo_label(X).predict(X)


    def predict_proba(self, X):
        return self._pseudo_label(X).predict_proba(X)


    def decision_function(self, X):
        return self._pseudo_label(X).decision_function(X)


    def _pseudo_label(self, X):

        while not self.max_iter or self.n_iter_ < self.max_iter:

            # Select not added rows
            index = X.index.difference(self.X_.index)
            X_new = X.loc[index]

            if not len(index):
                break

            # Predict probabilities
            y_prob = self.estimator_.predict_proba(X_new)
            y_prob = pd.DataFrame(y_prob, index=X_new.index)
            y_new = y_prob.apply(lambda row: row.argmax(), axis=1)

            # Mask rows with high certainty
            mask = (y_prob >= self.proba).any(axis=1)
            if not mask.any():
                break

            # Add labeled data & fit
            self.partial_fit(X_new[mask], y_new[mask])

            # Verbose
            if self.verbose:
                logmsg(f"ITER {self.n_iter_}: Add {mask.sum()} labels")

        return self.estimator_


    @if_delegate_has_method(delegate='estimator_')
    def score(self, X, y=None, *args, **kwargs):
        return self.estimator_.score(X, y, *args, **kwargs)

    #@if_delegate_has_method(delegate='estimator_')
    #def predict(self, X):
    #    return self.estimator_.predict(X)

    #@if_delegate_has_method(delegate='estimator_')
    #def predict_proba(self, X):
    #    return self.estimator_.predict_proba(X)

    #@if_delegate_has_method(delegate='estimator_')
    #def predict_log_proba(self, X):
    #    return self.estimator_.predict_log_proba(X)

    #@if_delegate_has_method(delegate='estimator_')
    #def decision_function(self, X):
    #    return self.estimator_.decision_function(X)

    def __getattr__(self, item):
        return getattr(self.estimator_, item)
