from sklearn.model_selection import ParameterSampler, ParameterGrid
from sklearn.base import BaseEstimator, clone

import optuna, hyperopt
import scipy

import time

from collections.abc import Iterable
from numbers import Number

import pandas as pd
import numpy as np

from ._output import plot_progress, print_progress

from ..crossval import crossval_score
#from ..model._model import MODEL_PARAMS, PREP_PARAMS, FIT_PARAMS





#__all__ = ['GridSearchCV', 'RandomSearchCV', 'OptunaCV']
#__all__ = ['BaseOptimizer', 'qround', 'ranking']



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

    param_space : dict
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

    max_time : int or NoneType (default: None)
        Stop optimization after given number of seconds.
        None means no time limitation. If n_trials is also set to None,
        the optimization continues to create trials until it receives a
        termination signal such as Ctrl+C or SIGTERM.

    max_trials : int or NoneType (default: None)
        Stop optimization after given number of evaluations (iterations).
        None means no iterations limitation. If timeout is also set to None,
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

    plot : bool, optional (default: False)
        Plot scores if True

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
    def __init__(self, estimator, cv=5, scoring=None, param_space=None,
                 max_time=None, max_trials=None, n_jobs=None,
                 verbose=1, plot=False):

        self.estimator = estimator
        self.param_space = param_space

        self.cv = cv
        self.scoring = scoring

        self.max_time = max_time
        self.max_trials = max_trials

        self.n_jobs = n_jobs

        self.verbose = verbose
        self.plot = plot



    def eval_params(self, params):
        '''Evaluate new trial and save to history DataFrame <trials_>.
        Can be called by user manually.

        Args
        ----
            params : dict
                New trial

        Return
        ------
            score : float or NoneType
                Returns trial's score (greater is always better). If status is
                not 'ok', returns None.

        '''
        if self.is_finished:
            return

        time_start = time.time()

        params = fix_params(params, self.param_space)

        trial = {
            'status': 'ok',
            'params': params,
            'score': None,
            'time': .0
        }

        try:
            # Check if <max_time> reached
            self._check_time()

            # Get estimator with new params
            estimator = self._get_estimator(params)

            # Fit & evaluate new estimator
            scores = self._eval(estimator)
            trial['score'] = np.mean(scores)

            # TODO: score for each fold & std
            # TODO: multiple scoring
            # TODO: ranking


        except KeyboardInterrupt:
            trial['status'] = 'interrupt'

            self.is_finished = True
            raise KeyboardInterrupt


        except TimeoutError:
            trial['status'] = 'timeout'

            self.is_finished = True
            raise KeyboardInterrupt


        except Exception as ex:
            trial['status'] = 'fail'

            print('[{}] {}'.format(type(ex).__name__, ex))


        finally:
            # Calculate time difference
            trial['time'] = time.time() - time_start

            # Save trial & show user
            self._save_trial(trial)
            self._output()

            return trial['score']



    def _save_trial(self, trial):
        '''Save new trial to history DataFrame <trials_> and update best.

        Args
        ----
            trial : dict
                Contains keys ['status', 'params', 'score', 'time']

        '''
        # Save new trial
        self.trials_ = self.trials_.append(trial, ignore_index=True)
        self.n_trials_ = len(self.trials_)

        # Check if last
        if self.n_trials_ >= self.max_trials:
            self.is_finished = True

        # Update best trial
        # (if better, than previous)
        if trial['status'] is 'ok':

            if self.best_score_ and trial['score'] > self.best_score_ \
            or self.best_score_ is None:

                self.best_trial_ = self.n_trials_ - 1
                self.best_score_ = self.trials_.loc[self.best_trial_, 'score']
                self.best_params_ = self.trials_.loc[self.best_trial_, 'params']

                self.best_estimator_ = self._get_estimator(self.best_params_)



    def _check_time(self):
        '''Check if <total_time_> less then <max_time> or if <max_time> is not
        defined (limited).

        Return
        ------
            time_left : float
                Time left (in seconds). Negative values means timeout reached.

        '''

        self.total_time_ = self.trials_['time'].sum() if self.n_trials_ else .0

        if self.max_time and (self.max_time < self.total_time_):
            raise TimeoutError



    def _get_estimator(self, params):
        '''Get estimator instance. Previous best if warm start. Clone if needed.

        Args
        ----
            params : dict
                Parameters to set

        Return
        ------
            estimator : estimator object
                The object to use to fit the data.

        '''

        # Continue optimization from previos best estimator
        if self.best_estimator_:
            estimator = clone(self.best_estimator_)
        else:
            estimator = clone(self.estimator)

        # Set params
        estimator = estimator.set_params(**params)

        return estimator



    def fit(self, X, y, groups=None):

        self._fit_start(X, y, groups)

        self._fit(X, y, groups) # defined individualy
        self._fit_end()

        return self



    def partial_fit(self, X, y, groups=None):

        self._fit_start(X, y, groups, partial_fit=True)

        self._fit(X, y, groups) # defined individualy
        self._fit_end()

        return self



    def _fit_start(self, X, y, groups=None, partial_fit=False):
        '''Starting routine. Initialize trials, space, starting time and define
        evaluator function (estimator -> score).

        Args
        ----
            X : DataFrame, shape [n_samples, n_features]
                The data to fit, score and calculate out-of-fold predictions

            y : Series, shape [n_samples]
                The target variable to try to predict

            groups : None
                Group labels for the samples used while splitting the dataset
                into train/test set

            partial_fit : boolean
                Whether to update existent trials

        '''

        # Init trials
        if not hasattr(self, 'trials_') \
        or partial_fit is False:

            # Reset trials
            self.trials_ = pd.DataFrame()
            self.n_trials_ = 0

            # Reset best
            self.best_trial_ = None
            self.best_score_ = None
            self.best_params_ = None

            self.best_estimator_ = None

        # Init data
        self.X_, self.y_, self.groups_ = X, y, groups

        # Init space
        self._init_space()

        # Check if <max_trials> is reached
        if self.max_trials and (self.max_trials > self.n_trials_) \
        or self.max_trials in [0, None]:
            self.is_finished = False
        else:
            self.is_finished = True



    def _eval(self, estimator):

        scores = crossval_score(estimator, self.cv, self.X_, self.y_, self.groups_,
                                self.scoring, n_jobs=self.n_jobs, verbose=0).values
        return scores



    def _fit_end(self):
        '''Fit ending routine: ranking, "is finished", ...
        '''
        if (self.trials_['status'] == 'ok').any():

            scores = self.trials_['score']
            self.trials_['rank'] = ranking(scores)



    def _init_space(self):
        '''Get boundaries types (btypes) and convert param_space to optimizer's
        specific format (space) via method <_get_space>.
        '''
        self.btypes = get_bound_types(self.param_space)
        self.space = self._get_space(self.param_space)



    def _output(self):
        '''Print verbose & plot output for the last trial.
        '''
        if self.plot:
            plot_progress(self)
        if self.verbose:
            print_progress(self)










'''class RandomSearchCV(Optimizer):

    def init_space(self):

        space = {}

        for param, btype in self.btypes.items():

            if btype is 'choice':
                space[param] = list(base_space[param])

            elif btype in ['uniform', 'quniform', 'quniform_int']:
                a, b = base_space[param][:2]
                space[param] = scipy.stats.uniform(a, b-a)

            elif btype is 'loguniform':
                a, b = base_space[param][:2]
                space[param] = scipy.stats.reciprocal(a, b)

            elif btype is 'const':
                pass

        self.space = space


    def optimize(self, n_trials=100, timeout=None, seed=0, **kwargs):

        self.set_limits(n_trials=n_trials, timeout=timeout)

        sampler = ParameterSampler(self.space, n_iter=n_trials, random_state=seed)

        try:
            for params in sampler:
                self.get_score(params)
        except:
            pass

        return self.best_params_'''





'''class DECV(Optimizer):

    def _init_space(self):

        space = {}

        for param, btype in self.btypes.items():

            if btype in ['uniform', 'quniform', 'quniform_int', 'loguniform']:
                a, b = self.base_space[param][:2]
                space[param] = (a, b)

            elif btype in ['const', 'choice']:
                pass

        self.space = space


    def optimize(self, n_trials=None, timeout=None, seed=0, **kwargs):

        self.set_limits(timeout=timeout)

        bounds = list(self.space.values())
        keys = list(self.space.keys())

        def objective(x):
            params = dict(zip(keys, x))
            score = self.get_score(params)
            return -score if score else None

        try:
            scipy.optimize.differential_evolution(objective, bounds, seed=seed, **kwargs)
        except:
            pass

        return self.best_params_'''





'''class HyperOptCV(Optimizer):

    def _init_space(self):

        space = {}

        for param, btype in self.btypes.items():

            if btype is 'choice':
                space[param] = hyperopt.hp.choice(param, self.base_space[param])

            elif btype is 'uniform':
                a, b = self.base_space[param]
                space[param] = hyperopt.hp.uniform(param, a, b)

            elif btype in ['quniform', 'quniform_int']:
                a, b, q = self.base_space[param]
                space[param] = hyperopt.hp.quniform(param, a, b, q)

            elif btype is 'loguniform':
                a, b = self.base_space[param][:2]
                a, b = np.log(a), np.log(b)
                space[param] = hyperopt.hp.loguniform(param, a, b)

            elif btype is 'const':
                pass

        self.space = space


    def optimize(self, n_trials=100, timeout=None, seed=0, algo='tpe', **kwargs):

        self.set_limits(n_trials=n_trials, timeout=timeout)

        def objective(params):
            score, score_std = self.get_score(params, return_std=True)
            if score:
                return {'status': 'ok', 'loss': -score, 'loss_variance': score_std**2}
            else:
                return {'status': 'fail'}

        if algo is 'tpe': algo = hyperopt.tpe.suggest
        elif algo is 'rand': algo = hyperopt.rand.suggest

        try:
            result = hyperopt.fmin(objective, self.space, algo=algo, max_evals=n_trials,
                rstate=np.random.RandomState(seed))
        except:
            pass

        return self.best_params_'''




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
