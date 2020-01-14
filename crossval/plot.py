from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring
from tqdm import tqdm_notebook as tqdm

import pandas as pd
import numpy as np

__all__ = [
    'plot_curve',
]

# ATTRS
STAGE_PARAM = {
    'CatBoost': 'ntree_end',
    'LGB': 'num_iteration',
    'XGB': 'ntree_limit',
}

class StagedPredictWrapper(BaseEstimator):

    def __init__(self, fitted_estimator, model='LGB', stage=100):
        self.fitted_estimator = fitted_estimator
        self.model = model
        self.stage = stage

    def predict_proba(self, X, **kwargs):
        kwargs = dict(kwargs)
        kwargs[self.stage_param_] = self.stage
        return self.fitted_estimator.predict_proba(X, **kwargs)

    def predict(self, X, **kwargs):
        kwargs = dict(kwargs)
        kwargs[self.stage_param_] = self.stage
        return self.fitted_estimator.predict(X, **kwargs)

    def score(self, X, y):
        return self.fitted_estimator.score(X, y)

    @property
    def _estimator_type(self):
        return self.fitted_estimator._estimator_type

    @property
    def stage_param_(self):
        return STAGE_PARAM[self.model]


def plot_curve(result, X, y, groups=None, model='LGB',
               stages=[10, 20, 30, 40, 50]):

    estimators = result['estimator']
    scorer = result['scorer']
    cv = result['cv']

    trn_scores = []
    val_scores = []

    # Get Staged Scores
    pbar = tqdm(total=cv.get_n_splits() * len(stages))

    for (trn, val), estimator in zip(cv.split(X, y, groups), estimators):

        X_trn, y_trn = X.iloc[trn], y.iloc[trn]
        X_val, y_val = X.iloc[val], y.iloc[val]

        trn_scores_ = []
        val_scores_ = []

        for stage in stages:
            estimator_ = StagedPredictWrapper(estimator, model, stage)
            trn_scores_.append(scorer(estimator_, X_trn, y_trn))
            val_scores_.append(scorer(estimator_, X_val, y_val))
            pbar.update(1)

        trn_scores.append(trn_scores_)
        val_scores.append(val_scores_)

    # Plot Learning Curve
    trn_scores = np.array(trn_scores)
    val_scores = np.array(val_scores)

    trn_avg = trn_scores.mean(axis=0)
    val_avg = val_scores.mean(axis=0)
    trn_std = trn_scores.std(axis=0)
    val_std = val_scores.std(axis=0)

    plt.fill_between(stages, trn_avg-trn_std, trn_avg+trn_std, alpha=.1, color='b')
    plt.fill_between(stages, val_avg-val_std, val_avg+val_std, alpha=.1, color='y')
    plt.plot(stages, trn_scores.mean(axis=0), label='train score', color='b')
    plt.plot(stages, val_scores.mean(axis=0), label='valid score', color='y')
    plt.legend()
    plt.show()
