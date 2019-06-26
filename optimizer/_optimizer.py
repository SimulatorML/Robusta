from sklearn.model_selection import ParameterSampler, ParameterGrid
from sklearn.base import BaseEstimator, clone

import optuna, hyperopt
import scipy

import time

from collections.abc import Iterable
from numbers import Number

import pandas as pd
import numpy as np

from ..crossval import crossval_score

from ._output import plot_progress, print_progress

#from ..model._model import MODEL_PARAMS, PREP_PARAMS, FIT_PARAMS





#__all__ = ['GridSearchCV', 'RandomSearchCV', 'OptunaCV']
__all__ = ['OptunaCV']



class BaseOptimizer(BaseEstimator):
    '''
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

    param_space : dict

    clone_estimator : boolean (default: True)

    return_estimator : boolean (default: False)
        If return_estimator=True, <fit> method returns estimator with optimal
        parameters (attr <best_estimator_>), otherwise return optimizer object.

    timeout : int or NoneType (default: None)
        Stop optimization after given number of seconds.
        None means no time limitation. If n_trials is also set to None,
        the optimization continues to create trials until it receives a
        termination signal such as Ctrl+C or SIGTERM.

    n_trials : int or NoneType (default: None)
        Stop optimization after given number of evaluations (iterations).
        None means no iterations limitation. If timeout is also set to None,
        the optimization continues to create trials until it receives a
        termination signal such as Ctrl+C or SIGTERM.

    warm_start : boolean (default: False)
        Whether to continue previous optimization process when calling <fit>.
        Otherwise delete old trials and start new optimization.

    random_state : int (default: 0)
        Random seed for stochastic oprimizers.

    verbose : int, optional (default: 1)
        Verbosity level:
        0: No output
        1: Print time, iters, score & eta
        2: Also print trial's parameters
        3: Also print cv output for each fold

    plot : bool, optional (defautl: False)
        Plot scores if True

    Attributes
    ----------

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

    best_estimator_ : estimator
        Estimator with best params

    best_score_ : float
        Best score

    best_params_ : dict
        Best parameters

    best_trial_ : int
        Index of best trial

    '''
    def __init__(self, estimator, cv=5, scoring=None, param_space=None,
                 clone_estimator=True, return_estimator=False, warm_start=False,
                 timeout=None, n_trials=None, random_state=0, debug=False,
                 verbose=1, plot=False):

        self.param_space = param_space

        self.estimator = estimator
        self.clone_estimator = clone_estimator
        self.return_estimator = return_estimator

        self.scoring = scoring
        self.cv = cv

        self.warm_start = warm_start
        self.timeout = timeout
        self.n_trials = n_trials

        self.verbose = verbose
        self.plot = plot

        self.random_state = random_state
        self.debug = debug



    def eval_params(self, params):

        if self.is_finished_:
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
            if self._time_left() < 0:
                raise TimeoutError

            # TODO: Do not clone for Pipeline
            estimator = clone(self.best_estimator_)
            estimator = estimator.set_params(**params)

            scores = self.eval(estimator)
            trial['score'] = np.mean(scores)
            # TODO: score for each fold & std
            # TODO: multiple scoring
            # TODO: ranking

        except KeyboardInterrupt:
            trial['status'] = 'interrupt'

            self.is_finished_ = True
            raise KeyboardInterrupt

        except TimeoutError:
            trial['status'] = 'timeout'

            self.is_finished_ = True
            raise KeyboardInterrupt


        except Exception as ex:
            trial['status'] = 'fail'

            if self.debug:
                print('[{}] {}'.format(type(ex).__name__, ex))


        finally:
            trial['time'] = time.time() - time_start

            self._save_trial(trial)
            self._output()

            return trial['score']



    def _save_trial(self, trial):

        self.trials_ = self.trials_.append(trial, ignore_index=True)


    def _init_trials(self):

        if not (self.warm_start and hasattr(self, 'trials_')):
            # Reset optimization
            # max trials = new trials
            self.trials_ = pd.DataFrame()
            self.max_trials = self.n_trials
            self.best_estimator_ = self.estimator

        elif self.n_trials is None:
            # Continue optimization (with no limits)
            # max trials = current trials + new trials
            self.max_trials = None

        else:
            # Continue optimization (with limits)
            # max trials = current trials + new trials
            self.max_trials = len(self.trials_) + self.n_trials



    def _set_eval(self, X, y, groups):

        self.eval = lambda estimator: crossval_score(estimator, self.cv,
            X, y, groups, self.scoring, n_jobs=-1, verbose=0).values



    def _time_left(self):

        time_delta = self.timeout if self.timeout else 0
        return self.time_start + time_delta - time.time()



    def _fit(self, X, y, groups):

        self._init_trials()
        self._init_space()

        self._set_eval(X, y, groups)
        self.is_finished_ = False

        self.time_start = time.time()



    def _fit_end(self):

        self.is_finished_ = True
        self.n_trials_ = len(self.trials_)

        if self.n_trials_ and (self.trials_['status'] == 'ok').any():

            best_trial = self.trials_['score'].argmax()

            self.best_trial_ = best_trial
            self.best_score_ = self.trials_.loc[best_trial, 'score']
            self.best_params_ = self.trials_.loc[best_trial, 'params']

            self.best_estimator_.set_params(**self.best_params_)



    def _init_space(self):

        self.btypes = get_bound_types(self.param_space)
        self.space = self._get_space()


    def _output(self):
        '''
        Print verbose & plot output for the last trial (self.trials_[-1]).
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





class OptunaCV(BaseOptimizer):

    def _get_space(self):
        return self.param_space


    def _get_params(self, trial):

        space = self.space
        params = {}

        for param, btype in self.btypes.items():

            if btype is 'choice':
                params[param] = trial.suggest_categorical(param, space[param])

            elif btype is 'uniform':
                a, b = space[param]
                params[param] = trial.suggest_uniform(param, a, b)

            elif btype is 'quniform':
                a, b, q = space[param]
                b = qround(b, a, b, q)
                params[param] = trial.suggest_discrete_uniform(param, a, b, q)

            elif btype is 'quniform_int':
                a, b = space[param][:2]
                params[param] = trial.suggest_int(param, a, b)

            elif btype is 'loguniform':
                a, b = space[param][:2]
                params[param] = trial.suggest_loguniform(param, a, b)

            elif btype is 'const':
                pass

        return params


    def fit(self, X, y, groups=None):

        self._fit(X, y, groups)

        optuna.logging.disable_default_handler()

        def objective(trial):

            params = self._get_params(trial)
            score = self.eval_params(params)

            if self.is_finished_:
                raise KeyboardInterrupt
            else:
                return -score if score else np.nan

        try:
            if not (self.warm_start and hasattr(self, 'study')):
                sampler = optuna.samplers.TPESampler(seed=self.random_state)
                self.study = optuna.create_study(sampler=sampler)
            self.study.optimize(objective, n_trials=self.n_trials)

        except KeyboardInterrupt:
            pass

        self._fit_end()

        if self.return_estimator:
            return self.best_estimator_
        else:
            return self




def get_bound_types(space):

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

                    elif bounds[2] == 1:
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

        - For quniform distribution: round param value with defined step
        - For constant: replace parameter's value with defined constant

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
    x = max(min(x, a), b)

    # Round x (with defined step q)
    x = a + ((x - a)//q)*q
    x = round(x, decimals)

    # Convert x to integer (if both a & q are integers)
    if isinstance(a + q, int):
        x = int(x)

    return x
