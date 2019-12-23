import numpy as np
import pandas as pd

from sklearn.utils.random import check_random_state
from deap import creator, base, tools, algorithms

from robusta.utils import logmsg, get_ranks
from .base import _GroupSelector, _WrappedSelector

from ._plot import _plot_progress, _plot_subset


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




class GeneticSelector(_WrappedSelector):
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

    def __init__(self, estimator, cv=5, scoring=None, n_gen=None, crossover='one',
                 min_features=0.1, max_features=0.9, pop_size=50, mutation=0.01,
                 max_time=None, random_state=None, n_jobs=None, verbose=1,
                 n_digits=4):

        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv

        self.min_features = min_features
        self.max_features = max_features

        self.crossover = crossover
        self.mutation = mutation

        self.max_time = max_time
        self.pop_size = pop_size
        self.n_gen = n_gen

        self.random_state = random_state
        self.verbose = verbose

        self.n_digits = n_digits
        self.n_jobs = n_jobs


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
            self.k_gen_ = 0

            # Init toolbox
            self.toolbox = base.Toolbox()
            self.rstate = check_random_state(self.random_state)

            # Define individual
            k_min = self.min_features_
            k_max = self.max_features_

            def get_individual():
                ind_size = self.rstate.choice(range(k_min, k_max+1))
                features = self.features_.sample(ind_size)
                return features

            self.toolbox.register("individual", get_individual)

            # Define population
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            self.population = self.toolbox.population(n=self.pop_size)

        return self


    def _fit(self, X, y, groups):

        # Define crossover & mutation
        mate = CROSSOVER[self.crossover]
        self.toolbox.register("mate", mate, random_state=self.rstate)
        self.toolbox.register("mutate", mutSubset, random_state=self.rstate, indpb=self.mutation)

        # Define evaluation & selection
        self.toolbox.register("eval", self.eval_subset, X=X, y=y, groups=groups)
        self.toolbox.register("select", tools.selTournament, tournsize=5, fit_attr='score')

        while not self.n_gen or self.k_gen_ < self.n_gen:

            if self.verbose:
                logmsg(f'GENERATION {self.k_gen_+1}')

            try:
                #offspring = [ind.copy() for ind in self.population]
                offspring = []

                # Apply crossover
                if self.k_gen_ > 0:
                    weights = [ind.score for ind in self.population]
                    weights = get_ranks(weights, normalize=True)
                else:
                    weights = None

                for _ in range(self.pop_size):
                    ind1, ind2 = self.rstate.choice(self.population, 2, p=weights)
                    child, _ = self.toolbox.mate(ind1, ind2)
                    offspring.append(child)

                # Apply mutation
                for ind in offspring:
                    self.toolbox.mutate(ind)

                # Evaluate
                for ind in offspring:
                    self.toolbox.eval(ind)

                # Select
                self.population = self.toolbox.select(offspring, k=self.pop_size)
                self.k_gen_ += 1

            except KeyboardInterrupt:
                break

            if self.verbose:
                print()

                scores = [ind.score for ind in offspring]
                avg = np.mean(scores)
                std = np.std(scores)

                logmsg('SCORE AVG: {:.{n}f} ± {:.{n}f}'.format(avg, std, n=self.n_digits))
                logmsg('SCORE MIN: {:.{n}f}'.format(np.min(scores), n=self.n_digits))
                logmsg('SCORE MAX: {:.{n}f}'.format(np.max(scores), n=self.n_digits))
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

    def plot_progress(self, **kwargs):
        kwargs_ = dict(marker='.', linestyle='--', alpha=0.3, c='g')
        kwargs_.update(kwargs)
        return _plot_progress(self, **kwargs_)

    def plot_subset(self, **kwargs):
        kwargs_ = dict(marker='.', linestyle='--', alpha=0.3, c='g')
        kwargs_.update(kwargs)
        return _plot_subset(self, **kwargs_)



class GroupGeneticSelector(_GroupSelector, GeneticSelector):
    pass
