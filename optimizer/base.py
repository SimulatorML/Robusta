from collections.abc import Iterable
from numbers import Number
from time import time
from typing import Union, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from . import _plot_progress
from . import _print_last
from ..crossval import crossval_score
from ..testing import extract_param_space


class BaseOptimizer(BaseEstimator):
    """
    Hyper-parameters Optimizer

    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data.

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

    scoring : string, callable or None, optional, default: None
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only a single value.
        If None, the estimator's default scorer (if available) is used.

    param_space : dict or None (default=None)
        Parameters bounds:

            - 'uniform':
                Uniform distribution [a, b].
                Pass 2 values (a, b) as tuple.

            - 'quniform':
                Uniform distribution [a, b] with step q.
                Pass 3 values (a, b, q) as tuple.

            - 'quniform_int':
                Uniform distribution [a, b] with integer step q=1.
                Pass 3 values (a, b, 1) as tuple of integers.

            - 'loguniform':
                Log-uniform distribution [log10(a), log10(b)].
                Pass 3 values (a, b, 'log') as tuple.

            - 'choice':
                Set of options A, B, C & etc.
                Pass several values {A, B, C, ...} as set.

            - 'const':
                Constant value.
                Pass single value (int, float, string, None, ...).

        If <param_space> set to None, use automatic parameters setting.

    warm_start : bool (default: False)
        If True, continue optimization after last <fit> call. If False, reset
        trials history and start new optimization.

        Warning: if last <fit> call ended with iteration / time limit exceed,
        you should change them before new <fit> call.

    max_time : int or NoneType (default: None)
        Stop optimization after given number of seconds.
        None means no time limitation. If <max_iter> is also set to None,
        the optimization continues to create trials until it receives a
        termination signal such as Ctrl+C or SIGTERM.

    max_iter : int or NoneType (default: None)
        Stop optimization after given number of evaluations (iterations).
        None means no iterations limitation. If <max_time> is also set to None,
        the optimization continues to create trials until it receives a
        termination signal such as Ctrl+C or SIGTERM.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel (for inner cross-validation).
        None means 1.

    verbose : int, optional (default: 1)
        Verbosity level:
        0: No output
        1: Print time, iters, score & eta
        2: Also print trial's parameters
        3: Also print cv output for each fold

    n_digits : int (default=4)
        Verbose score(s) precision

    Attributes
    ----------

    X_, y_, groups_ : ndarray-like, array-like, array-like
        Last fit data

    best_estimator_ : estimator
        Estimator with best params

    trials_ : DataFrame
        Params, score, time & cv results:

        - 'params' : dict
            Estimated parameters

        - 'score', 'std' : float
            Mean score of cross-validation & it's standard deviation

        - 'time' : float
            Fitting duration (# of sec)

        - 'status': string
            Final status of trial ('ok', 'timeout', 'fail' or 'interrupted')

    n_trials_ : int
        Total number of trials

    best_score_ : float
        Best score

    best_params_ : dict
        Best parameters

    best_trial_ : int
        Index of best trial

    total_time_ : float
        Total optimization time

    """

    def __init__(self,
                 estimator: BaseEstimator,
                 cv: int = 5,
                 scoring: Optional[Union[str, Callable]] = None,
                 param_space: Optional[dict] = None,
                 warm_start: bool = False,
                 max_time: Optional[int] = None,
                 max_iter: Optional[int] = None,
                 n_jobs: Optional[int] = None,
                 verbose: int = 1,
                 n_digits: int = 4,
                 debug: bool = False):

        self.estimator = estimator
        self.param_space = param_space
        self.warm_start = warm_start

        self.cv = cv
        self.scoring = scoring

        self.max_time = max_time
        self.max_iter = max_iter

        self.verbose = verbose
        self.n_digits = n_digits
        self.n_jobs = n_jobs
        self.debug = debug

    def eval_params(self,
                    params: dict,
                    X: pd.DataFrame,
                    y: pd.Series,
                    groups: pd.Series = None) -> Union[float]:
        """
        Evaluates the given hyperparameters on the provided dataset using cross-validation.

        Parameters
        ----------
        params : dict
            The hyperparameters to evaluate.
        X : pd.DataFrame
            The input features for the dataset.
        y : pd.Series
            The output target values for the dataset.
        groups : pd.Series, optional
            The group labels for the dataset. Default is None.

        Returns
        -------
        Union[float, np.nan]
            The mean score obtained through cross-validation. If an exception is raised during the evaluation
            or the evaluation is interrupted by the user, np.nan is returned.
        """

        # Check if the maximum number of iterations has been exceeded
        self._check_max_iter()

        # Check if the maximum time has been exceeded
        self._check_max_time()

        # Record the start time of the evaluation
        tic = time()

        try:
            # Fix the hyperparameters to ensure they are in the expected format
            params = fix_params(params, self.param_space_)

            # Create a new estimator with the given hyperparameters
            estimator = clone(self.estimator).set_params(**params)

            # Perform cross-validation and compute scores
            scores = crossval_score(estimator=estimator,
                                    cv=self.cv,
                                    X=X,
                                    y=y,
                                    groups=groups,
                                    scoring=self.scoring,
                                    n_jobs=self.n_jobs,
                                    verbose=0)

            # Record the results of the trial
            trial = {
                'params': params,
                'status': 'ok',
                'time': time() - tic,
                'score': np.mean(scores),
                'score_std': np.std(scores),
                'scores': scores,
            }

            # Add the trial to the list of trials
            self._append_trial(trial)

            # Print the results of the trial
            _print_last(self)

            # Return the mean score obtained through cross-validation
            return trial['score']

        except KeyboardInterrupt:
            # If the user interrupts the evaluation, raise a KeyboardInterrupt exception
            raise KeyboardInterrupt

            # Record the results of the trial as failed
            trial = {
                'params': params,
                'status': 'fail',
                'time': time() - tic,
            }

            # Add the trial to the list of trials
            self._append_trial(trial)

            # Print the results of the trial
            _print_last(self)

            # Return np.nan to indicate that the evaluation was interrupted
            return np.nan

        except (Exception,):
            # If an exception is raised during the evaluation, and debug mode is on, re-raise the exception
            if self.debug:
                raise

    def _append_trial(self,
                      trial: dict) -> None:
        """
        Append a trial to the trials_ dataframe.

        Parameters
        ----------
        trial : dict
            Trial from the optimizer.

        Returns
        -------
        Nothing:
            None
        """
        self.trials_ = self.trials_.append(trial, ignore_index=True)

    @property
    def best_iter_(self) -> int:
        """
        Return the index of the best trial.

        Returns
        -------
        best_iter: int
            Best iteration from trials
        """
        return self.trials_['score'].idxmax()

    @property
    def best_score_(self) -> int:
        """
        Return best score

        Returns
        -------
        score : int
            Best score among trials
        """
        return self.trials_['score'][self.best_iter_]

    @property
    def best_params_(self) -> dict:
        """
        Return the parameter settings of the best trial.

        Returns
        -------
        params : dict
            Best parameter settings
        """
        return self.trials_['params'][self.best_iter_]

    @property
    def best_estimator_(self) -> BaseEstimator:
        """
        Return a new estimator object fit with the best parameters.

        Returns
        -------
        self:
            BaseEstimator
        """
        return clone(self.estimator).set_params(**self.best_params_)

    @property
    def n_iters_(self) -> int:
        """
        Number of iterations performed during the optimization process.

        Returns
        -------
        int:
            Number of iterations.
        """
        return len(self.trials_) if hasattr(self, 'trials_') else 0

    @property
    def total_time_(self) -> float:
        return self.trials_['time'].sum() if hasattr(self, 'trials_') else .0

    @property
    def predict(self) -> float:
        """
        Total time spent during the optimization process.

        Returns
        -------
        float:
            Total time spent (in seconds).
        """
        return self.trials_['time'].sum() if hasattr(self, 'trials_') else .0

    def _check_max_iter(self) -> None:
        """
        Checks if the maximum number of iterations has been reached and raises a KeyboardInterrupt if it has.
        """
        if hasattr(self, 'max_iter') and self.max_iter:
            if self.max_iter <= self.n_iters_:
                if self.verbose: print('Iterations limit exceed!')
                raise KeyboardInterrupt

    def _check_max_time(self) -> None:
        """
        Checks if the maximum time limit has been reached and raises a KeyboardInterrupt if it has.
        """
        if hasattr(self, 'max_time') and self.max_time:
            if self.max_time <= self.total_time_:
                if self.verbose: print('Time limit exceed!')
                raise KeyboardInterrupt

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            groups: pd.Series = None) -> 'BaseOptimizer':
        """
        Fits the optimizer to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Training input samples.
        y : pd.Series
            Target values.
        groups : pd.Series, default=None
            Group labels for the samples.

        Returns
        -------
        self:
            Returns an instance of self.
        """

        # Check if params set to auto
        self.param_space_ = self.param_space
        if not self.param_space_:
            self.param_space_ = extract_param_space(self.estimator, verbose=self.verbose)

        # Define new space
        if not self.warm_start or not hasattr(self, 'btypes'):
            self.btypes = get_bound_types(self.param_space_)
            self.space = self._get_space(self.param_space_)

        # Reset trials
        if not self.warm_start or not hasattr(self, 'trials_'):
            self.trials_ = pd.DataFrame()

        # Custom core fit
        self._fit(X, y, groups)

        return self

    def plot(self,
             **kwargs) -> None:
        """
        Plots the optimization process using the defined plotting function.

        Parameters
        ----------
        **kwargs: dict
            Additional keyword arguments to pass to the plotting function.
        """
        _plot_progress(self, **kwargs)


def get_bound_types(space: dict) -> dict:
    """
    Get parameter's type
        - 'uniform': uniform distribution [a, b]
        - 'quniform': uniform distribution [a, b] with step q
        - 'quniform_int': uniform distribution [a, b] with integer step q
        - 'loguniform': log-uniform distribution [log10(a), log10(b)]
        - 'choice' : set of options {A, B, C, ...}
        - 'const': any single value

    Parameters
    ----------
    space : dict
        Boundaries

    Returns
    -------
        btypes : dict
            Boundaries type
    """
    btypes = {}

    for param, bounds in space.items():

        if isinstance(bounds, str):
            btype = 'const'

        elif isinstance(bounds, Iterable):
            if isinstance(bounds, set):
                btype = 'choice'

            elif isinstance(bounds, tuple):

                if len(bounds) == 2:
                    btype = 'uniform'

                elif len(bounds) == 3:

                    if bounds[2] == 'log':
                        btype = 'loguniform'

                    elif isinstance(bounds[2], int):
                        btype = 'quniform_int'

                    elif isinstance(bounds[2], Number):
                        btype = 'quniform'

                    else:
                        raise ValueError('Unknown bounds type: {}'.format(bounds))

                else:
                    raise ValueError('Unknown bounds type: {}'.format(bounds))

            else:
                raise ValueError('Unknown bounds type: {}'.format(bounds))

        else:
            btype = 'const'

        btypes[param] = btype

    return btypes


def fix_params(params: dict,
               space: dict) -> dict:
    """
    Normalize parameters value according to defined space:

        - 'quniform': round param value with defined step
        - 'constant': replace parameter's value with defined constant

    Parameters
    ----------
    params : dict
        Parameters

    space : dict
        Boundaries

    Returns
    -------
    fixed_params : dict
        Normalized parameters
    """

    # Create a copy of the parameter dictionary
    params = dict(params)

    # Determine the boundary types for each parameter in the space
    btypes = get_bound_types(space)

    # Normalize the parameters values according to the boundary types
    for param, bounds in space.items():

        if btypes[param] in ['quniform', 'quniform_int']:
            a, b, q = bounds
            params[param] = qround(params[param], a, b, q)

        elif btypes[param] is 'const':
            params[param] = bounds

    return params


def ranking(ser: pd.Series) -> pd.Series:
    """
    Make rank transformation.

    Parameters
    ----------
    ser : Series
        Values for ranking. None interpreted as worst.

    Returns
    -------
    rnk : Series of int
        Ranks (1: highest, N: lowest)

    """
    ser = ser.fillna(ser.min())

    rnk = ser.rank(method='dense', ascending=False)
    rnk = rnk.astype(int)

    return rnk


def qround(x: Union[int, float],
           a: Union[int, float],
           b: Union[int, float],
           q: Union[int, float],
           decimals: int = 4):
    """
    Convert x to one of [a, a+q, a+2q, .., b]

    Parameters
    ----------
    x : int or float
        Input value. x must be in [a, b].
        If x < a, x set to a.
        If x > b, x set to b.

    a, b : int or float
        Boundaries. b must be greater than a. Otherwize b set to a.

    q : int or float
        Step value. If q and a are both integer, x set to integer too.

    decimals : int, optional (default: 4)
        Number of decimal places to round to.


    Returns
    -------
    x_new : int or float
        Rounded value

    """
    # Check if a <= x <= b
    b = max(a, b)
    x = min(max(x, a), b)

    # Round x (with defined step q)
    x = a + ((x - a) // q) * q
    x = round(x, decimals)

    # Convert x to integer
    if isinstance(a + q, int):
        x = int(x)

    return x
