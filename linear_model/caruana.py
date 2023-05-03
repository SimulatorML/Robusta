import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import get_scorer
from tqdm import tqdm_notebook as tqdm


class _BaseCaruana(LinearModel):
    """
    Caruana Ensemble Selection for Regression/Classification

    Parameters
    ----------

    scoring : str
        Objective for optimization.

    iters : int (default=100)
        Number of models in ensemble.

    init_iters : int (default=10)
        Number of core models in ensemble, which selected from whole set
        of models at the beginning. Values from range 5-25 are prefered.

        Set 0 for basic algorithm.

    colsample : float or int (default=0.5)
        Number of models, sampled on each iteration. Must be from range (0, 1].

        Set 1.0 for basic algorithm.

    replace : bool (default=True)
        Whether to reuse models, already added to the ensemble (recommended).

        Set False for basic algorithm.

    random_state : int, RandomState instance, or None (default=None)
        Pseudo-random number generator to control the subsample of models.

    verbose : int (default=1)
        Verbosity level.

    n_jobs : int or None (default=None)
        The number of jobs to use for the computation.
        `None` means 1. `-1` means using all processors.

    tqdm : bool (default=False)
        Whether to show progress bar.

    Attributes
    ----------
    weights_ : list of int
        Number of times each model was used.

    y_avg_ : float
        Target bias
    """

    def __init__(
        self,
        scoring: str,
        iters: int = 100,
        init_iters: int = 10,
        colsample: float = 0.5,
        replace: bool = True,
        random_state: int = None,
        n_jobs: int = -1,
        tqdm: bool = False,
    ):
        self.y_avg_ = None
        self.weights_ = None
        self.scorer = None
        self.classes_ = None
        self.iters = iters
        self.init_iters = init_iters
        self.scoring = scoring
        self.colsample = colsample
        self.replace = replace
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tqdm = tqdm

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_BaseCaruana":
        """
        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            Stacked predictions.

        y : DataFrame or Series, shape [n_samples, ] or [n_samples, n_classes]
            Target variable

        Returns
        -------
        self
        """

        # Check if the model is a classifier
        if self._estimator_type is "classifier":
            self.classes_ = np.unique(y)

        # Get the scoring function for the objective
        self.scorer = get_scorer(self.scoring)

        # Initialize the weights and target bias
        self.weights_ = np.zeros(X.shape[1])
        self.y_avg_ = y.mean()

        # Check the initialization parameters
        msg = "<init_iters> must be no more than <iters>"
        assert self.init_iters <= self.iters, msg

        if not self.replace:
            msg = "<iters> must be no more than X.shape[1] (if replace=True)"
            assert self.iters <= X.shape[1], msg

        # Initial subset
        scores = {}
        for k in range(X.shape[1]):
            self.weights_[k] += 1
            scores[k] = self.score(X, y)
            self.weights_[k] -= 1

        scores = pd.Series(scores).sort_values(ascending=False)
        scores = scores[: self.init_iters]
        self.weights_[scores.index] += 1

        # Core Algorithm
        i_range = range(self.init_iters, self.iters)
        if self.tqdm:
            i_range = tqdm(i_range, initial=self.init_iters, total=self.iters)

        for _ in i_range:
            k_range = np.arange(X.shape[1])

            if not self.replace:
                k_range = k_range[self.weights_ == 0]

            if self.colsample < 1.0:
                p = 1 + int(len(k_range) * self.colsample)
                k_range = np.random.choice(k_range, p, replace=False)

            best_score = None
            best_k = -1

            for k in k_range:
                self.weights_[k] += 1
                score = self.score(X, y)
                self.weights_[k] -= 1

                if best_k < 0 or best_score < score:
                    best_score = score
                    best_k = k

            self.weights_[best_k] += 1

        return self

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Returns the score of the blended model on the given data.

        Parameters
        ----------
        X : pd.DataFrame
            Features data for prediction.
        y : pd.Series
            Target data for prediction.

        Returns
        -------
        float:
            The score of the blended model on the given data.
        """
        return self.scorer(self, X, y)

    def _blend(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns the predicted values of the blended model on the given data.

        Parameters
        ----------
        X : pd.DataFrame
            Features data for prediction.

        Returns
        -------
        np.ndarray:
            The predicted values of the blended model on the given data.
        """
        return X.dot(self.coef_).values + self.intercept_

    @property
    def coef_(self) -> np.ndarray:
        """
        Returns the weights of each model in the blend.

        Returns
        -------
        np.ndarray:
            The weights of each model in the blend.
        """
        if self.weights_.any():
            return np.array(self.weights_) / np.sum(self.weights_)
        else:
            return self.weights_

    @property
    def intercept_(self) -> float:
        """
        Returns the intercept value of the blended model.

        Returns
        -------
        float:
            The intercept value of the blended model.
        """
        return 0.0 if self.coef_.any() else self.y_avg_


class CaruanaRegressor(_BaseCaruana, RegressorMixin):
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for a given input.

        Parameters
        -----------
        X: np.ndarray
            The input data to make predictions on. The shape should be (n_samples, n_features).

        Returns
        --------
        predictions: np.ndarray
            The predicted labels for the input data. The shape should be (n_samples,).
        """
        return self._blend(X)


class CaruanaClassifier(_BaseCaruana, ClassifierMixin):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for a given input.

        Parameters
        -----------
        X: np.ndarray
            The input data to make predictions on. The shape should be (n_samples, n_features).

        Returns
        --------
        proba: np.ndarray
            A 2D array of shape (n_samples, 2), where each row represents the probability of the
            corresponding sample belonging to each class. The first column is the probability of
            the negative class, and the second column is the probability of the positive class.
        """
        y = self._blend(X)
        return np.stack([1 - y, y], axis=-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for a given input.

        Parameters
        -----------
        X: np.ndarray
            The input data to make predictions on. The shape should be (n_samples, n_features).

        Returns
        --------
        predictions: np.ndarray
            The predicted labels for the input data. The shape should be (n_samples,).
            Each label is either 0 (for the negative class) or 1 (for the positive class).
        """
        y = self.predict_proba(X)
        return np.rint(y[:, 1]).astype(int)
