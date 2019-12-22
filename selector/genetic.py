import numpy as np
import pandas as pd

from sklearn.utils.random import check_random_state
from deap import creator, base, tools, algorithms

from robusta.utils import logmsg
from .base import _GroupSelector, _AgnosticSelector

from ._plot import _plot_progress


__all__ = ['GeneticSelector', 'GroupGeneticSelector']




def cxUniform(ind1, ind2, indpb=0.5, random_state=None, drop_attrs=['score']):

    rstate = check_random_state(random_state)
    mask1, mask2 = [], []

    for x, y in zip(ind1.mask, ind2.mask):
        if rstate.rand() < indpb:
            mask1.append(x)
            mask2.append(y)
        else:
            mask1.append(y)
            mask2.append(x)

    child1 = ind1.copy().set_mask(mask1)
    child2 = ind2.copy().set_mask(mask2)

    child1.parents = (ind1, ind2)
    child2.parents = (ind1, ind2)

    for attr in drop_attrs:
        for child in [child1, child2]:
            if hasattr(child, attr):
                delattr(child, attr)

    return child1, child2


def cxOnePoint(ind1, ind2, indpb=0.5, random_state=None, drop_attrs=['score']):

    rstate = check_random_state(random_state)

    n = ind1.n_features
    argsort = rstate.permutation(n)

    a = rstate.randint(n)

    mask1 = np.zeros((n,), dtype=bool)
    mask2 = np.zeros((n,), dtype=bool)

    for i in range(n):
        j = argsort[i]
        x = ind1.mask[i]
        y = ind2.mask[j]
        if a <= i:
            mask1[j] = x
            mask2[j] = y
        else:
            mask1[j] = y
            mask2[j] = x

    child1 = ind1.copy().set_mask(mask1)
    child2 = ind2.copy().set_mask(mask2)

    child1.parents = (ind1, ind2)
    child2.parents = (ind1, ind2)

    for attr in drop_attrs:
        for child in [child1, child2]:
            if hasattr(child, attr):
                delattr(child, attr)

    return child1, child2


def cxTwoPoint(ind1, ind2, indpb=0.5, random_state=None, drop_attrs=['score']):

    rstate = check_random_state(random_state)

    n = ind1.n_features
    argsort = rstate.permutation(n)

    a = rstate.randint(n)
    b = rstate.randint(n)
    a, b = min(a, b), max(a, b)

    mask1 = np.zeros((n,), dtype=bool)
    mask2 = np.zeros((n,), dtype=bool)

    for i in range(n):
        j = argsort[i]
        x = ind1.mask[i]
        y = ind2.mask[j]
        if a <= i <= b:
            mask1[j] = x
            mask2[j] = y
        else:
            mask1[j] = y
            mask2[j] = x

    child1 = ind1.copy().set_mask(mask1)
    child2 = ind2.copy().set_mask(mask2)

    child1.parents = (ind1, ind2)
    child2.parents = (ind1, ind2)

    for attr in drop_attrs:
        for child in [child1, child2]:
            if hasattr(child, attr):
                delattr(child, attr)

    return child1, child2


CROSSOVER = {
    'one': cxOnePoint,
    'two': cxTwoPoint,
    'uni': cxUniform,
}




def mutSubset(ind, indpb, random_state=None, drop_attrs=['score']):

    rstate = check_random_state(random_state)
    mask = []

    for x in ind.mask:
        y = (rstate.rand() < indpb)
        mask.append(x ^ y)

    mutant = ind.set_mask(mask)

    for attr in drop_attrs:
        if hasattr(mutant, attr):
            delattr(mutant, attr)

    return mutant




class GeneticSelector(_AgnosticSelector):
    '''Feature Selector based on Differential Evolution algorithm

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.

    cv : int, cross-validation generator or iterable
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.

    scoring : string, callable or None, optional, default: None
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only a single value.
        If None, the estimator's default scorer (if available) is used.

    mut_freq : float [0..1], default=0.1
        Percentage of mutants in population

    mut_prob : float [0..1], default=0.05
        Probability of mutation in single cell

    crossover : float [0..1], default=0.5
        Proportion of recombination of recombinations in population

    pop_size : int, default=20
        Population size (number of individuals)

    max_iter : int or None
        Maximum number of iterations. None for no limits. Use <max_time>
        or Ctrl+C for KeyboardInterrupt to stop optimization in this case.

    max_time : float or None
        Maximum time (in seconds). None for no limits. Use <max_iter>
        or Ctrl+C for KeyboardInterrupt to stop optimization in this case.

    random_state : int or None (default=0)
        Random seed for permutations in PermutationImportance.
        Ignored if <importance_type> set to 'inbuilt'.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel. None means 1.

    verbose : int, optional (default=1)
        Verbosity level

    n_digits : int (default=4)
        Verbose score(s) precision

    '''

    def __init__(self, estimator, cv=5, scoring=None, cx_type='one', cx_rate=0.5,
                 mut_freq=0.1, mut_prob=0.05, max_iter=None, max_time=None,
                 pop_size=20, ind_init=0.5, random_state=None, n_jobs=None,
                 verbose=1, n_digits=4):

        self.estimator = estimator
        self.scoring = scoring
        #self.std = std
        self.cv = cv

        self.cx_type = cx_type
        self.cx_rate = cx_rate

        self.mut_freq = mut_freq
        self.mut_prob = mut_prob

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.max_time = max_time

        self.ind_init = ind_init

        self.random_state = random_state
        self.n_jobs = n_jobs

        self.verbose = verbose
        self.n_digits = n_digits


    def fit(self, X, y, groups=None):

        self._fit_start(X)
        self._fit(X, y, groups)

        return self


    def partial_fit(self, X, y, groups=None):

        self._fit_start(X, partial=True)
        self._fit(X, y, groups)

        return self


    def _fit_start(self, X, partial=False):

        self._set_features(X)

        if not partial or not hasattr(self, 'trials_'):

            self._reset_trials()
            self.n_gen_ = 0

            # Init toolbox
            self.toolbox = base.Toolbox()
            self.rstate = check_random_state(self.random_state)

            # Define individual
            self.toolbox.register("individual", self.features_.sample)

            # Define population
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            self.population = self.toolbox.population(n=self.pop_size)

        return self


    def _fit(self, X, y, groups):

        # Define crossover
        crossover = CROSSOVER[self.cx_type]
        self.toolbox.register("mate", crossover, random_state=self.rstate)

        # Define evaluation, mutation, selection
        self.toolbox.register("eval", self.eval_subset, X=X, y=y, groups=groups)
        self.toolbox.register("mutate", mutSubset, random_state=self.rstate, indpb=self.mut_prob)
        self.toolbox.register("select", tools.selTournament, tournsize=3, fit_attr='score')

        while not self.max_iter or self.n_iters_ < self.max_iter:

            self.n_gen_ += 1

            if self.verbose:
                logmsg(f'GENERATION {self.n_gen_}')

            try:
                # Select the next generation individuals
                offspring = [ind.copy() for ind in self.population]

                # Apply crossover
                for i in range(1, len(offspring), 2):
                    if self.rstate.rand() < self.cx_rate:
                        parent1, parent2 = offspring[i-1:i+1]
                        child1, child2 = self.toolbox.mate(parent1, parent2)
                        offspring[i-1:i+1] = child1, child2

                # Apply mutation
                for mutant in offspring:
                    if self.rstate.rand() < self.mut_freq:
                        self.toolbox.mutate(mutant)

                # Evaluate
                for ind in offspring:
                    if not hasattr(ind, 'score') and len(ind):
                        self.toolbox.eval(ind)

                # Select
                self.population = self.toolbox.select(offspring, k=self.pop_size)

            except KeyboardInterrupt:
                break

            if self.verbose:
                print()

                score = [ind.score for ind in offspring]
                avg = np.mean(score)
                std = np.std(score)

                logmsg('SCORE AVG: {:.{n}f} ± {:.{n}f}'.format(avg, std, n=self.n_digits))
                logmsg('SCORE MIN: {:.{n}f}'.format(np.min(score), n=self.n_digits))
                logmsg('SCORE MAX: {:.{n}f}'.format(np.max(score), n=self.n_digits))
                print()

                sizes = [ind.n_selected for ind in offspring]
                avg = int(np.mean(sizes))
                std = np.std(sizes)

                logmsg('SIZE AVG: {} ± {:.{n}f}'.format(avg, std, n=self.n_digits))
                logmsg('SIZE MIN: {}'.format(np.min(sizes)))
                logmsg('SIZE MAX: {}'.format(np.max(sizes)))
                print()

        return self


    def get_subset(self):
        return self.best_subset_

    def plot(self, **kwargs):
        kwargs_ = dict(marker='.', linestyle='--', alpha=0.3, c='g')
        kwargs_.update(kwargs)
        return _plot_progress(self, **kwargs_)



class GroupGeneticSelector(_GroupSelector, GeneticSelector):
    pass
