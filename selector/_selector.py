import pandas as pd
import numpy as np

from itertools import combinations
from functools import reduce
import operator as op
import datetime

from sklearn.utils.random import check_random_state

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll import scope

from robusta import utils
from ._output import *


__all__ = ['SFM', 'RFE', 'SFS', 'RFS']



class Selector():

    def __init__(self, cv, model, model_params={}, prep_params={},
                 fit_params={}, use_cols=None, **fs_params):

        self.cv, self.model = cv.copy(), model.copy()

        self.model_params = model_params
        self.prep_params = prep_params
        self.fit_params = fit_params
        self.fs_params = fs_params

        self.use_cols = set(use_cols) if use_cols else set(cv.X_train.columns)
        self.max_cols = len(self.use_cols)

        self.reset_trials()



    def get_score(self, cols, prev=[], return_importance=False,
                  save_trial=True):

        cols = set(cols)
        prev = set(prev) if prev else []

        trial = {
            'cols': cols,
            'prev': prev,
            'score': None,
            'importance': None,
            'time': None,
        }

        same_trial = self.find_trial(cols)

        if same_trial:
            trial.update(same_trial)

        else:
            args = (self.model, self.model_params, self.prep_params, self.fit_params, cols)
            cv = self.cv.copy().fit(*args)

            trial['importance'] = cv.importance()
            trial['score'] = cv.score()
            trial['time'] = cv.time()

        if save_trial:
            self.save_trial(**trial)

        if return_importance:
            return trial['score'], trial['importance']
        else:
            return trial['score']



    def save_trial(self, cols, score, prev=[], importance=None, time=None,
                   **kwargs):

        trial = {
            'score': score,
            'cols': set(cols),
            'prev': set(prev),
            'n_cols': len(cols),
            'n_prev': len(prev),
            'importance': importance,
            'time': time,
        }

        self.trials_.append(trial)

        # Update Best
        if not self.best_score_ or (score and score > self.best_score_):
            self.best_score_ = score
            self.best_cols_ = cols
            self.best_iter_ = len(self.trials_)-1

        # Output
        self.output()


    def find_trial(self, cols):

        if not cols:
            return

        mask = (np.array(self.get_key_trials('cols')) == set(cols))

        if mask.any():
            trial = np.array(self.trials_)[mask][-1]
            return trial


    def reset_trials(self):
        self.trials_ = []
        self.best_score_ = None
        self.best_iter_ = None
        self.best_cols_ = None


    def get_key_trials(self, key='cols'):
        return [trial[key] for trial in self.trials_]


    def _get_cols(self, cols=None, forward=False):
        if cols is None:
            if forward:
                return set()
            else:
                return set(self.use_cols)
        else:
            return set(cols) & set(self.use_cols)



    def _get_k(self, k):

        if isinstance(k, float):
            assert 0 < k <= 1, 'k_cols float is out of range (0, 1].'
            k = int(np.ceil(k * self.max_cols))

        elif isinstance(k, int):
            msg = 'k_cols int is out of range [1, {}].'.format(self.max_cols)
            assert 1 <= k <= self.max_cols, msg

        else:
            raise TypeError('k_cols type must be int or float.')

        return k


    def _get_k_minmax(self, k_cols):

        if isinstance(k_cols, tuple):
            k_min, k_max = k_cols
            assert k_min <= k_max, 'min k_cols value must be smaller than max k_cols value.'

        elif isinstance(k_cols, int) or isinstance(k_cols, float):
            k = self._get_k(k_cols)
            k_min, k_max = k_cols, k_cols

        return k_min, k_max


    def set_output(self, plot=False, forward=False, verbose=1):

        self.cv.plot = False
        self.cv.verbose = 2 if (int(verbose) > 2) else 0

        self.plot = plot
        self.verbose = verbose
        self.forward = forward



    def output(self):
        if self.plot:
            plot_progress(self)
        if self.verbose:
            print_progress(self)




class SFM(Selector):
    '''
    Select From Model
    '''
    def select(self, k_cols, base_cols=None, plot=False, verbose=True):

        self.set_output(verbose=verbose, plot=plot)

        # cols & k_cols
        cols, k_cols = self._get_cols(base_cols), self._get_k(k_cols)

        # Features rating
        _, imp = self.get_score(cols, return_importance=True)

        # Select k best & score
        best = utils.select_k_max(imp, k_cols)
        cols, prev = set(best.index), cols
        _ = self.get_score(cols, prev=prev)

        self.kept_cols = cols
        return list(self.kept_cols)



class RFE(Selector):
    '''
    Recursive Feature Elimination
    '''
    def select(self, k_cols, base_cols=None, plot=False, verbose=True):

        self.set_output(verbose=verbose, plot=plot)

        # feature selector parameters
        fs_params = {'base_cols': None, 'step': 1}
        for key in fs_params:
            if key in self.fs_params:
                setattr(self, key, self.fs_params[key])
            else:
                setattr(self, key, fs_params[key])

        # cols & k_cols
        cols, k_cols = self._get_cols(self.base_cols), self._get_k(k_cols)

        # Initial features importances
        _, cols_rate = self.get_score(cols, return_importance=True)

        while len(cols) > k_cols:

            # Drop N worse features (N == step)
            cols_update = cols_rate.sort_values().iloc[:self.step].index
            cols, prev = cols - set(cols_update.index), set(cols)

            # Update features importances
            _, cols_rate = self.get_score(cols, return_importance=True, prev=prev)

        return list(cols)



class SFS(Selector):
    '''
    Sequential Features Selector
        - (SFS) Sequential Forward Selection
        - (SBS) Sequential Backward Selection
        - (SFFS) Sequential Forward Floating Selection
        - (SBFS) Sequential Backward Floating Selection

    Parameters
    ----------
    forward : bool (default: True)
        Forward selection if True (SFS/SFFS), backward selection
        otherwise (SBS/SBFS).

    floating : bool (default: False)
        Adds a conditional exclusion/inclusion if True (SFFS/SBFS,
        otherwise SFS/SBS).

    max_subset : int, float, string or None (default: 'log2')
        The maximum number of random candidate features to select to check:

            - If int, then consider select 'min(max_subset, max_cols)' features.
            - If float, then 'max_subset' is a fraction and
            'int(max_subset * max_cols)' features are selected.
            - If "sqrt", then 'max_subset=sqrt(max_cols)'.
            - If "log2", then 'max_subset=log2(max_cols)'.
            - If "auto", then 'max_subset=log2(max_cols)'.
            - If "full", then 'max_subset=max_cols'.
            - If None, then 'max_subset=max_cols'.

    base_cols : set (or list) of string or None (default: None)
        Initial features subset to start from.

    random_state : int (default: 0)

    '''
    def select(self, k_cols, plot=False, verbose=True):

        # feature selector parameters
        fs_params = {
            'forward': True,
            'floating': False,
            'max_subset': 'log2', # TODO: current random subspace selection is not reproducible
            'base_cols': None,
            'random_state': 0, # TODO: currently reproducible only for the same instance of notebook
        }
        for key in fs_params:
            if key in self.fs_params:
                setattr(self, key, self.fs_params[key])
            else:
                setattr(self, key, fs_params[key])

        self.set_output(verbose=verbose, plot=plot, forward=self.forward)

        # cols & n_cols
        cols, k_cols = self._get_cols(self.base_cols, self.forward), self._get_k(k_cols)

        # subset max size
        if self.max_subset in ['full', None]:
            self.max_subset = self.max_cols

        elif self.max_subset in ['sqrt', 'auto']:
            self.max_subset = int(1+np.sqrt(self.max_cols))

        elif self.max_subset in ['log2']:
            self.max_subset = int(1+np.log2(self.max_cols))

        elif isinstance(self.max_subset, float):
            self.max_subset = int(1+ self.max_subset * self.max_cols)

        elif isinstance(self.max_subset, int):
            pass

        else:
            raise ValueError('Unknown max_subset value type: {}'.format(self.max_subset))

        # final criterion (step 1)
        # start criterion (step 2)
        if self.forward:
            is_final = lambda cols: len(cols) >= k_cols
            is_start = lambda cols: len(cols) == 1
        else:
            is_final = lambda cols: len(cols) <= k_cols
            is_start = lambda cols: len(cols) == self.max_cols
            self.get_score(cols)

        # init random state
        random_state_ = check_random_state(self.random_state)

        while not is_final(cols):

            # Step 1. Step Forward/Backward
            if self.forward:
                cols_updates = self.use_cols - cols
            else:
                cols_updates = set(cols)

            # Random subspace
            cols_updates = random_state_.permutation(list(cols_updates))[:self.max_subset]

            # Score gain
            cols_score = pd.Series(None, index=cols_updates)

            for col in cols_updates:

                # include/exclude (trial)
                if self.forward:
                    candidate = cols | {col}
                else:
                    candidate = cols - {col}
                score = self.get_score(candidate, prev=cols)
                cols_score[col] = score

            # select 1 best
            cols_update = utils.select_k_max(cols_score, 1)
            col = cols_update.index[0]
            old_score = cols_update[col]

            # include/exclude (final)
            if self.forward:
                cols = cols | {col}
            else:
                cols = cols - {col}

            # stop criteria
            if not self.floating or is_final(cols) or is_start(cols):
                continue

            # Step 2. Step Backward/Forward
            if self.forward:
                cols_updates = set(cols)
            else:
                cols_updates = self.use_cols - cols

            # Random subspace
            cols_updates = random_state_.permutation(list(cols_updates))[:self.max_subset]

            # Score gain
            cols_score = pd.Series(None, index=cols_updates)

            for col in cols_updates:

                # exclude/include (trial)
                if self.forward:
                    candidate = cols - {col}
                else:
                    candidate = cols | {col}

                # escaping loop
                if candidate in self.get_key_trials('cols'):
                    continue
                score = self.get_score(candidate, prev=cols)
                cols_score[col] = score

            # select best
            if sum(cols_score.notnull()) == 0:
                continue
            cols_update = utils.select_k_max(cols_score, 1)
            col = cols_update.index[0]
            new_score = cols_update[col]

            # is step back required?
            if new_score > old_score:
                # exclude/include (final)
                if self.forward:
                    cols = cols - {col}
                else:
                    cols = cols | {col}

        self.kept_cols = cols
        return list(self.kept_cols)



class RFS(Selector):
    '''
    Random Feature Selector
    '''
    def select(self, k_cols, n_iters, plot=False, verbose=True):

        self.set_output(verbose=verbose, plot=plot, forward=True)

        # feature selector parameters
        fs_params = {'random_state': 0}
        for key in fs_params:
            if key in self.fs_params:
                setattr(self, key, self.fs_params[key])
            else:
                setattr(self, key, fs_params[key])


        # init random state
        random_state_ = check_random_state(self.random_state)

        # range & weights
        k_range = self._get_k_minmax(k_cols)
        weights = utils.nCk_range(k_min=k_range[0], k_max=k_range[1], n=self.max_cols)

        for i in range(n_iters):

            # random subset (of random size) selection & evaluation
            k_cols = utils.weighted_choice(weights)
            cols = random_state_.choice(list(self.use_cols), k_cols, replace=False)
            score = self.get_score(cols)

        self.kept_cols = self.best_cols_
        return list(self.kept_cols)
