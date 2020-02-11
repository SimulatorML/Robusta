from sklearn.base import BaseEstimator
import numpy as np


class _StagedClassifier(BaseEstimator):

    _estimator_type = 'classifier'

    def predict_proba(self, X):
        return X

    def predict(self, X):
        y = X[:, 1] > 0.5
        y = y.astype(int)
        return y


class _StagedRegressor(BaseEstimator):

    _estimator_type = 'regressor'

    def predict(self, X):
        return X


def _xgb_staged_predict(estimator, X, max_iter=0, step=1):

    booster = estimator.get_booster()

    leafs = estimator.apply(X, max_iter)
    M = leafs.shape[0]
    N = leafs.shape[1]

    trees = np.mgrid[0:M, 0:N][1]
    return


def _lgb_staged_predict(estimator, X, max_iter=0, step=1):

    booster = estimator.booster_

    max_iter = max_iter if max_iter else booster.num_trees()

    leafs = booster.predict(X, pred_leaf=True, num_iteration=max_iter)
    M = leafs.shape[0]
    N = leafs.shape[1]

    trees = np.mgrid[0:M, 0:N][1]

    mapper = {}
    for i in range(estimator.n_estimators):
        for j in range(estimator.num_leaves):
            mapper[i, j] = booster.get_leaf_output(i, j)

    preds = np.vectorize(lambda i, j: mapper[i, j])(trees, leafs).T
    preds = preds.cumsum(axis=0)[np.arange(step, max_iter+step, step)-1]

    if estimator._estimator_type == 'classifier':
        preds = sigmoid(preds)

        for pred in preds:
            yield np.vstack([1-pred, pred]).T

    elif estimator._estimator_type == 'regressor':
        for pred in preds:
            yield pred


def _cat_staged_predict(estimator, X, max_iter=0, step=1):

    if estimator._estimator_type == 'classifier':
        return estimator.staged_predict_proba(X, ntree_end=max_iter, eval_period=step)

    elif estimator._estimator_type == 'regressor':
        return estimator.staged_predict(X, ntree_end=max_iter, eval_period=step)


def _get_scores(estimator, generator, predictor, trn, val, X, y,
                scorer, max_iter, step, train_score):

    stages = np.arange(step, max_iter+step, step)

    trn_scores = []
    val_scores = []

    if train_score:
        X_trn, y_trn = X.iloc[trn], y.iloc[trn]
        S_trn = generator(estimator, X_trn, max_iter, step)

        for _ in stages:
            trn_scores.append(scorer(predictor, next(S_trn), y_trn))

    if True:
        X_val, y_val = X.iloc[val], y.iloc[val]
        S_val = staged_predict(estimator, X_val, max_iter, step)

        for _ in stages:
            val_scores.append(scorer(predictor, next(S_val), y_val))

    return trn_scores, val_scores


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
