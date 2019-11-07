import numpy as np
import pandas as pd

from sklearn.utils.random import check_random_state
from deap import creator, base, tools, algorithms

from robusta.utils import logmsg
from .base import _GroupSelector, _AgnosticSelector


__all__ = ['GeneticSelector', 'GroupGeneticSelector']




def cxSubset(ind1, ind2, indpb=0.5, random_state=None):

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

    return child1, child2




def mutSubset(ind, indpb, random_state=None):

    rstate = check_random_state(random_state)
    mask = []

    for x in ind.mask:
        y = (rstate.rand() < indpb)
        mask.append(x ^ y)

    return ind.set_mask(mask)




class GeneticSelector(_AgnosticSelector):

    def __init__(self, estimator, cv=5, scoring=None, mut_prob=0.1, mut_rate=0.05,
                 crossover=0.5, pop_size=20, max_time=None, max_iter=None, std=-0.1,
                 random_state=None, n_jobs=None, verbose=1, n_digits=4):

        self.estimator = estimator
        self.scoring = scoring
        self.std = std
        self.cv = cv

        self.crossover = crossover
        self.mut_prob = mut_prob
        self.mut_rate = mut_rate
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.max_time = max_time

        self.random_state = random_state
        self.n_jobs = n_jobs

        self.verbose = verbose
        self.n_digits = n_digits


    def fit(self, X, y):

        self._fit_start(X)
        self._fit(X, y)

        return self


    def partial_fit(self, X, y):

        self._fit_start(X, partial=True)
        self._fit(X, y)

        return self


    def _fit_start(self, X, partial=False):

        self._set_features(X)

        if not partial or not hasattr(self, 'trials_'):

            self._reset_trials()

            # Init toolbox
            self.toolbox = base.Toolbox()
            self.rstate = check_random_state(self.random_state)

            # Define individual
            self.toolbox.register("individual", self.features_.sample)

            # Define population
            self.toolbox.register("population", tools.initRepeat,
                                  list, self.toolbox.individual)
            self.population = self.toolbox.population(n=self.pop_size)

        return self


    def _fit(self, X, y):

        # Define mutation & selection
        self.toolbox.register("eval", self._eval_subset, X=X, y=y)
        self.toolbox.register("mate", cxSubset, random_state=self.rstate)
        self.toolbox.register("mutate", mutSubset, random_state=self.rstate,
                              indpb=self.mut_rate)
        self.toolbox.register("select", tools.selTournament, tournsize=3,
                              fit_attr='score')

        self.n_gen_ = 0
        while not self.max_iter or self.n_iters_ < self.max_iter:

            self.n_gen_ += 1
            if self.verbose:
                logmsg('GENERATION {}'.format(self.n_gen_))

            try:
                # Select the next generation individuals
                offspring = [ind.copy() for ind in self.population]

                # Apply crossover
                for i in range(1, len(offspring), 2):
                    if self.rstate.rand() < self.crossover:
                        parent1, parent2 = offspring[i-1], offspring[i]
                        child1, child2 = self.toolbox.mate(parent1, parent2)
                        # FIXME:
                        if hasattr(child1, 'score'):
                            del child1.score, child1.score_std
                        if hasattr(child2, 'score'):
                            del child2.score, child2.score_std
                        offspring[i-1], offspring[i] = child1, child2

                # Apply mutation
                for mutant in offspring:
                    if self.rstate.rand() < self.mut_prob:
                        self.toolbox.mutate(mutant)
                        # FIXME:
                        if hasattr(mutant, 'score'):
                            del mutant.score, mutant.score_std

                # Evaluate
                for ind in offspring:
                    if not hasattr(ind, 'score') and len(ind):
                        self.toolbox.eval(ind)

                # Select
                self.population = self.toolbox.select(offspring, k=self.pop_size)

            except KeyboardInterrupt:
                break

        return self


    def get_subset(self):
        return self.best_subset_



class GroupGeneticSelector(_GroupSelector, GeneticSelector):
    pass
