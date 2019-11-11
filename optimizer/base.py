from sklearn.model_selection import ParameterSampler, ParameterGrid
from sklearn.base import BaseEstimator, clone

import optuna, hyperopt
import scipy

from time import time

from collections.abc import Iterable
from numbers import Number

import pandas as pd
import numpy as np

from ._verbose import _print_last
from ._plot import _plot_progress

from robusta.testing import extract_param_space
from robusta.crossval import crossval_score




class BaseOptimizer(BaseEstimator):
    '''Hyper-parameters Optimizer

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

    '''
    def __init__(self, estimator, cv=5, scoring=None, param_space='auto',
                 warm_start=False, max_time=None, max_iter=None, n_jobs=None,
                 verbose=1, n_digits=4):

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



    def eval_params(self, params, X, y, groups=None):

        self._check_max_iter()
        self._check_max_time()

        tic = time()

        params = fix_params(params, self.param_space)
        estimator = clone(self.estimator).set_params(**params)

        try:
            scores = crossval_score(estimator, self.cv, X, y, groups,
                                    self.scoring, n_jobs=self.n_jobs,
                                    verbose=0)

            trial = {
                'params': params,
                'status': 'ok',
                'time': time() - tic,
                'score': np.mean(scores),
                'score_std': np.std(scores),
                'scores': scores,
            }

            self._append_trial(trial)
            _print_last(self)

            return trial['score']

        except KeyboardInterrupt:
            raise KeyboardInterrupt

            trial = {
                'params': params,
                'status': 'fail',
                'time': time() - tic,
            }

            self._append_trial(trial)
            _print_last(self)

            return np.nan



    def _append_trial(self, trial):
        self.trials_ = self.trials_.append(trial, ignore_index=True)



    @property
    def best_iter_(self):
        return self.trials_['score'].idxmax()

    @property
    def best_score_(self):
        return self.trials_['score'][self.best_iter_]

    @property
    def best_params_(self):
        return self.trials_['params'][self.best_iter_]

    @property
    def best_estimator_(self):
        return clone(self.estimator).set_params(**self.best_params_)

    @property
    def n_iters_(self):
        return len(self.trials_) if hasattr(self, 'trials_') else 0

    @property
    def total_time_(self):
        return self.trials_['time'].sum() if hasattr(self, 'trials_') else .0



    def _check_max_iter(self):
        if hasattr(self, 'max_iter') and self.max_iter:
            if self.max_iter <= self.n_iters_:
                if self.verbose: print('Iterations limit exceed!')
                raise KeyboardInterrupt



    def _check_max_time(self):
        if hasattr(self, 'max_time') and self.max_time:
            if self.max_time <= self.total_time_:
                if self.verbose: print('Time limit exceed!')
                raise KeyboardInterrupt



    def fit(self, X, y, groups=None):

        # Check if params set to auto
        param_space = self.param_space
        if not param_space == 'auto':
            param_space = extract_param_space(self.estimator, verbose=self.verbose)

        # Define new space
        if not self.warm_start or not hasattr(self, 'btypes'):
            self.btypes = get_bound_types(param_space)
            self.space = self._get_space(param_space)

        # Reset trials
        if not self.warm_start or not hasattr(self, 'trials_'):
            self.trials_ = pd.DataFrame()

        # Custom core fit
        self._fit(X, y, groups)

        return self


    def plot(self, **kwargs):
        _plot_progress(self, **kwargs)






def get_bound_types(space):
    '''
    Get parameter's type

        - 'uniform': uniform distribution [a, b]
        - 'quniform': uniform distribution [a, b] with step q
        - 'quniform_int': uniform distribution [a, b] with integer step q
        - 'loguniform': log-uniform distribution [log10(a), log10(b)]
        - 'choice' : set of options {A, B, C, ...}
        - 'const': any single value

    Args
    ----
        space : dict
            Boundaries


    Returns
    -------
        btypes : dict
            Boundaries type
    '''
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

                    elif bounds[2] == 1 and isinstance(bounds[0], int):
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




def fix_params(params, space):
    '''
    Normalize parameters value according to defined space:

        - 'quniform': round param value with defined step
        - 'constant': replace parameter's value with defined constant

    Args
    ----
        params : dict
            Parameters

        space : dict
            Boundaries


    Returns
    -------
        fixed_params : dict
            Normalized parameters
    '''
    params = dict(params)
    btypes = get_bound_types(space)

    for param, bounds in space.items():

        if btypes[param] in ['quniform', 'quniform_int']:
            a, b, q = bounds
            params[param] = qround(params[param], a, b, q)

        elif btypes[param] is 'const':
            params[param] = bounds

    return params




def ranking(ser):
    '''Make rank transformation.

    Args
    ----
        ser : Series of float
            Values for ranking. None interpreted as worst.

    Returns
    -------
        rnk : Series of int
            Ranks (1: highest, N: lowest)

    '''
    ser = ser.fillna(ser.min())

    rnk = ser.rank(method='dense', ascending=False)
    rnk = rnk.astype(int)

    return rnk



def qround(x, a, b, q, decimals=4):
    '''
    Convert x to one of [a, a+q, a+2q, .., b]

    Args
    ----
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

    '''
    # Check if a <= x <= b
    b = max(a, b)
    x = min(max(x, a), b)

    # Round x (with defined step q)
    x = a + ((x - a)//q)*q
    x = round(x, decimals)

    # Convert x to integer
    if isinstance(a + q, int):
        x = int(x)

    return x
