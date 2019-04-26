from sklearn.model_selection import ParameterSampler, ParameterGrid
import optuna, hyperopt
import scipy

import time
import traceback, sys

from multiprocessing import TimeoutError
from collections.abc import Iterable
from numbers import Number
from funcy import project

import pandas as pd
import numpy as np

from ._output import plot_progress, print_progress

from robusta.model._model import MODEL_PARAMS, PREP_PARAMS, FIT_PARAMS
from robusta import utils


__all__ = ['RandomSearch', 'HyperOpt', 'Optuna', 'DE']



class Optimizer():
    '''
    Hyper-Parameters Optimizer

    Parameters
    ----------
    cv, model : instance
        CrossValidation & Model instances

    model_params, prep_params, fit_params : dict, optional (default: {})
        Fixed model & preparation & fit parameters. Also can be used as custom bounds.

    use_cols : iterable, optional (default: None)
        List of features names to use

    mode : string, optional
        'model': [default]
            Optimize ''model_params''
        'prep':
            Optimize ''prep_params''
        'fit':
            Optimize ''fit_params''

    verbose : int, optional (default: 1)
        Verbosity level:
        0: No output
        1: Print time, iters, score & eta
        2: Same plus trial's parameters
        3: Same plus cv output for each fold

    plot : bool, optional (defautl: False)
        Plot scores if True

    debug : bool, optional (defautl: False)
        Print error messages if True

    Attributes
    ----------

    best_params_ : dict
        Best parameters

    best_score_ : float
        Best score

    best_trial_ : int
        Index of best trial

    params_ : list of dicts
        All estimated parameters

    scores_ : list of float
        All scores

    trials_ : list of dicts
        All trials. Each trial contains:

        'params' : dict
            Estimated parameters

        'score', 'std' : float
            Mean score of cross-validation & it's standard deviation

        'time' : float
            Fitting duration (# of sec)

        'status': string
            Final status of trial ('ok', 'timeout', 'fail' or 'interrupted')

    time_ : float
        Total optimization time

    '''
    def __init__(self, cv, model, model_params={}, prep_params={}, fit_params={},
                 use_cols=None, mode='model', verbose=1, plot=False, debug=False):

        self.cv, self.model = cv.copy(), model.copy()

        self.model_params = model_params
        self.prep_params = prep_params
        self.fit_params = fit_params
        self.use_cols = use_cols


        # Space selection
        modes = {'model', 'prep', 'fit'}
        if mode in modes:
            self.mode = mode
        else:
            raise ValueError("Unknown mode: '{}'. Expected: {}.".format(mode, modes))

        if self.mode is 'model':
            base_space = dict(MODEL_PARAMS[model.model_name])
            base_space.update(self.model_params)

        elif self.mode is 'prep':
            base_space = dict({})
            base_space.update(PREP_PARAMS[self.prep_params])

        elif self.mode is 'fit':
            base_space = dict(FIT_PARAMS[model.model_name])
            base_space.update(self.fit_params)


        # Init output & space & trials
        self.set_output(plot, verbose, debug)
        self.set_space(base_space)
        self.reset_trials()



    def get_score(self, params, save_trial=True, return_std=False):

        if self.is_finished:
            return

        trial = {
            'params': self.fix_params(params),
            'score': None,
            'std': None,
            'status': 'interrupted',
            'time': .0
        }

        time_start = utils.ctime()

        try:
            kwargs = {
                'model': self.model,
                'model_params': dict(self.model_params),
                'prep_params': dict(self.prep_params),
                'fit_params': dict(self.fit_params),
                'use_cols': self.use_cols,
            }

            kwargs['{}_params'.format(self.mode)] = trial['params']

            cv = self.cv.copy().fit(**kwargs)
            trial['score'], trial['std'] = cv.score(return_std=True)

            trial.update({'status': 'ok'})

        except KeyboardInterrupt:
            trial.update({'status': 'interrupted'})
            self.is_finished = True
            raise KeyboardInterrupt

        #except TimeoutError:
        #    trial.update({'status': 'timeout'})

        except Exception as ex:
            trial.update({'status': 'fail'})

            if self.debug:
                print('[{}] {}'.format(type(ex).__name__, ex))

        finally:
            time_end = utils.ctime()
            time_delta = (time_end-time_start).total_seconds()
            trial['time'] = time_delta

            self.save_trial(**trial)
            self.output()

            if return_std:
                return trial['score'], trial['std']
            else:
                return trial['score']



    def save_trial(self, params, score=None, std=None, time=None, status='ok'):

        trial = {
            'params': params,
            'score': score,
            'std': std,
            'time': time,
            'status': status,
        }

        # Update best
        if score:
            if not self.best_score_ or score >= self.best_score_:
                self.best_params_ = params
                self.best_score_ = score
                self.best_trial_ = len(self.trials_)

        # Append new trial
        self.trials_.append(trial)
        self.params_.append(trial['params'])
        self.scores_.append(trial['score'])

        # Update optimization time
        self.time_ += trial['time']
        if self.max_time and self.time_ > self.max_time:
            self.is_finished = True


    def reset_trials(self):

        self.trials_ = []
        self.params_ = []
        self.scores_ = []

        self.best_params_ = None
        self.best_score_ = None
        self.best_trial_ = None

        self.is_finished = False
        self.time_ = .0


    def get_key_trials(self, key='params'):
        return [trial[key] for trial in self.trials_]


    def init_base_space(self, base_space):

        space = {}

        for param, bounds in base_space.items():

            # dependent bounds: bounds = f(x, y, ...)
            if callable(bounds):
                #args = {arg: dict_args[arg] for arg in utils.get_params(bounds)}
                args = {arg: getattr(self, arg) for arg in utils.get_params(bounds)}
                bounds = bounds(**args)

            # list bounds: [x, y, ...]
            if isinstance(bounds, list):

                for i in range(len(bounds)):
                    sub_param = '{}@{}'.format(param, i)
                    space[sub_param] = bounds[i]

            # regular bounds
            else:
                space[param] = bounds

        self.base_space = space


    def init_btypes(self):

        btypes = {}

        for param, bounds in self.base_space.items():
            # Step 3. Other bounds typification
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

        self.btypes = btypes


    def set_space(self, base_space):

        self.init_base_space(base_space)
        self.init_btypes()
        self.init_space()


    def fix_params(self, params):

        params = dict(params)

        # Normalize:
        # - integers
        # - constant
        for param, bounds in self.base_space.items():

            if self.btypes[param] in ['quniform', 'quniform_int']:
                a, b, q = bounds
                params[param] = utils.round_step(params[param], q=q, a=a)

            elif self.btypes[param] is 'const':
                params[param] = bounds


        # Squeeze lists
        base_params = set([param.split('@')[0] for param in params])

        for param in base_params:
            if param not in params:

                params[param] = []
                i = 0

                while '{}@{}'.format(param, i) in params:
                    sub_param = '{}@{}'.format(param, i)
                    params[param].append(params[sub_param])
                    del params[sub_param]
                    i += 1

        return params


    def set_limits(self, n_trials=None, timeout=None):
        '''
        Set iterations limits

        Parameters
        ----------
        n_trials : int or None (default: None)
            The maximum number of trials. If self.trials is not empty, then n_iters
            adds to previous max_iters.

        timeout : int, float or None (default: None)
            Stop optimization after given time (# of sec). If self.trials is not empty,
            then timeout adds to previous max_time.

        '''
        if n_trials:
            if len(self.trials_):
                self.max_trials = n_trials + len(self.trials_)
            else:
                self.max_trials = n_trials
        else:
            self.max_trials = None

        if timeout:
            if len(self.trials_):
                self.max_time = timeout + sum(self.get_key_trials('time'))
            else:
                self.max_time = timeout
        else:
            self.max_time = None


    def set_output(self, plot=True, verbose=1, debug=False):
        '''
        Output setting.

        Parameters
        ----------
        verbose : int

        plot : bool

        debug : bool
        '''
        self.cv.verbose = 2 if (int(verbose) > 2) else 0
        self.cv.plot = False

        self.verbose = verbose
        self.debug = debug
        self.plot = plot


    def output(self):
        '''
        Print verbose & plot output for the last trial (self.trials[-1]).
        '''
        if self.plot:
            plot_progress(self)
        if self.verbose:
            print_progress(self)






class RandomSearch(Optimizer):

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

        return self.best_params_





class DE(Optimizer):

    def init_space(self):

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

        return self.best_params_





class HyperOpt(Optimizer):

    def init_space(self):

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

        return self.best_params_





class Optuna(Optimizer):

    def init_space(self):
        self.space = self.base_space


    def get_params(self, trial):

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
                b = utils.round_step(b, q=q, a=a)
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


    def optimize(self, n_trials=None, timeout=None, seed=0, algo='tpe', **kwargs):

        self.set_limits(n_trials=n_trials, timeout=timeout)

        def objective(trial):
            params = self.get_params(trial)
            score = self.get_score(params)
            if self.is_finished: raise KeyboardInterrupt
            return -score if score else np.nan

        if algo is 'tpe': sampler = optuna.samplers.TPESampler(seed=seed)
        elif algo is 'rand': sampler = optuna.samplers.RandomSampler(seed=seed)

        try:
            optuna.logging.disable_default_handler()
            self.study = optuna.create_study(sampler=sampler)
            self.study.optimize(objective, n_trials=n_trials, timeout=timeout)
        except KeyboardInterrupt:
            pass

        return self.best_params_
