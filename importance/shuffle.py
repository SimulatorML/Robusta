from typing import Union, List, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
)
from sklearn.utils.random import check_random_state
from tqdm import tqdm_notebook as tqdm

from . import get_importance
from .. import crossval


def _shuffle_data(*data: Union[pd.DataFrame, pd.Series],
                  seed: int) -> List[Union[pd.DataFrame, pd.Series]]:
    """
    Shuffle the data along axis 0, maintaining the same order for all data arrays.

    Parameters
    ----------
    *data : Union[pd.DataFrame, pd.Series]
        A variable number of pandas DataFrames or Series to be shuffled.
    seed: int
        The random seed to use for the shuffling.

    Returns
    -------
    data_ :
        A list containing the shuffled data arrays.
    """

    # Get the index of the first data array
    x_index = data[0].index
    data_ = []

    # Iterate over each data array
    for x in data:
        # Check if the data array is not None
        if x is not None:
            # Shuffle the data array along axis 0 using the specified random seed
            x = x.sample(frac=1, random_state=seed)
            # Set the index of the shuffled data array to match the index of the first data array
            x.index = x_index
        # Append the shuffled data array to the output list
        data_.append(x)

    # Return the list of shuffled data arrays
    return data_


class ShuffleTargetImportance(BaseEstimator, MetaEstimatorMixin):
    """
    Shuffle Target importance for feature evaluation.

    Parameters
    ----------
    estimator : object
        The base estimator. This can be both a fitted
        (if ``prefit`` is set to True) or a non-fitted estimator.

    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

    scoring : string, callable or None, default=None
        Scoring function to use for computing feature importances.
        A string with scoring name (see scikit-learn docs) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    n_repeats : int, default 5
        The number of random shuffle iterations. Decrease to improve speed,
        increase to get more precise estimates.

    mode : {'dif', 'div'}, default='dif'
        How to calculate mode between importances after shuffling and benchmark.

        - 'dif': for difference between importances (absolute mode).
        - 'div': for division between importances (relative mode).

    tqdm : bool, default=False
        Whether to display <tqdm_notebook> progress bar while iterating
        through out dataset columns.

    verbose : int, default=0
        Verbosity level

    n_jobs : int, default -1
        The number of jobs to run in parallel. None means 1.

    random_state : integer or numpy.random.RandomState, optional
        Pseudo-random number generator to control the permutations of each feature.

    cv_kwargs : dict
        Key arguments for inner crossval function

    Attributes
    ----------
    feature_importances_ : Series, shape (n_groups, )
        Feature importances, computed as mean decrease of the importance when
        a target is shuffled (i.e. becomes noise).

    feature_importances_std_ : Series, shape (n_groups, )
        Standard deviations of feature importances.

    raw_importances_ : list of Dataframes, shape (n_folds, n_groups, n_repeats)

    scores_ : list of floats, shape (n_folds, )

    """

    def __init__(self,
                 estimator: BaseEstimator,
                 cv: int,
                 scoring: Optional[Union[str, Callable]] = None,
                 n_repeats=5,
                 mode: str = 'dif',
                 tqdm: bool = False,
                 verbose: int = 0,
                 n_jobs: Optional[int] = None,
                 random_state: Optional[int] = None,
                 cv_kwargs: Optional[dict] = None):
        self.feature_importances_std_ = None
        self.feature_importances_ = None
        self.shuff_scores_ = None
        self.scores_ = None
        self.bench_scores_ = None
        self.raw_importances_ = None
        self.shuff_importances_ = None
        self.bench_importances_ = None
        if cv_kwargs is None:
            cv_kwargs = {}
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.cv_kwargs = cv_kwargs
        self.n_repeats = n_repeats
        self.mode = mode
        self.tqdm = tqdm
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            groups: np.array = None) -> 'ImportanceEstimator':
        """
        Fits the ImportanceEstimator to the input data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the estimator.
        y: pd.Series
            The target variable.
        groups: np.array, optional
            Group labels for the samples.

        Returns
        -------
        self:
            ImportanceEstimator, the fitted estimator.

        """

        rstate = check_random_state(self.random_state)

        # Check that n_repeats is a positive integer
        msg = "<n_repeats> must be positive integer"
        assert isinstance(self.n_repeats, int) and self.n_repeats > 0, msg

        # Check that mode is either 'dif' or 'div'
        msg = "<mode> must be in {'dif', 'div'}"
        assert self.mode in ['dif', 'div'], msg

        # Initialize lists to store feature importances and scores
        self.bench_importances_ = []
        self.shuff_importances_ = []
        self.raw_importances_ = []
        self.bench_scores_ = []
        self.shuff_scores_ = []
        self.scores_ = []

        # Iterate over the number of repeats
        iters = range(self.n_repeats)
        iters = tqdm(iters) if self.tqdm else iters

        for _ in iters:
            # Shuffle the data
            seed = rstate.randint(2 ** 32 - 1)
            X_, y_, groups_ = _shuffle_data(X, y, groups, seed=seed)

            # Benchmark
            result = crossval(estimator=self.estimator,
                              cv=self.cv,
                              X=X_,
                              y=y_,
                              groups=groups_,
                              scoring=self.scoring,
                              verbose=self.verbose,
                              return_estimator=True,
                              return_pred=False,
                              n_jobs=self.n_jobs,
                              **self.cv_kwargs)

            # Store the feature importances and scores
            for e in result['estimator']:
                self.bench_importances_.append(get_importance(e) + 1)
            self.scores_.append(np.mean(result['val_score']))
            self.bench_scores_.append(result['val_score'])

            # Shuffle the target variable
            result = crossval(estimator=self.estimator,
                              cv=self.cv,
                              X=X,
                              y=y_,
                              groups=groups,
                              scoring=self.scoring,
                              verbose=self.verbose,
                              return_estimator=True,
                              return_pred=False,
                              n_jobs=self.n_jobs,
                              **self.cv_kwargs)

            # Store the feature importances and scores
            for e in result['estimator']:
                self.shuff_importances_.append(get_importance(e) + 1)
            self.shuff_scores_.append(result['val_score'])

        # Compute the raw feature importances
        for b, s in zip(self.bench_importances_, self.shuff_importances_):
            if self.mode == 'dif': self.raw_importances_.append(b - s)
            if self.mode == 'div': self.raw_importances_.append(b / s)

        # Compute the average feature importances and their standard deviation
        imps = self.raw_importances_
        self.feature_importances_ = np.average(imps, axis=0)
        self.feature_importances_std_ = np.std(imps, axis=0)

        return self
