from typing import Optional, Tuple, List, Callable, Union

import numpy as np
import pandas as pd
from deap import base, tools
from sklearn.base import BaseEstimator
from sklearn.utils.random import check_random_state

from . import _WrappedGroupSelector, _WrappedSelector
from . import _plot_progress, _plot_subset
from ..utils import logmsg, get_ranks, secfmt


def cxUniform(ind1: base.Toolbox,
              ind2: base.Toolbox,
              indpb: float = 0.5,
              random_state: Optional[int] = None,
              drop_attrs: Optional[list] = None) -> Tuple[base.Toolbox, base.Toolbox]:
    """
    Perform a uniform crossover between two individuals.

    Parameters
    ----------
    ind1 : base.Toolbox
        The first individual to be crossed.
    ind2 : base.Toolbox
        The second individual to be crossed.
    indpb : float, optional
        The probability of an attribute being swapped between the two individuals. Defaults to 0.5.
    random_state : int, optional
        Seed for the random number generator. Defaults to None.
    drop_attrs : list, optional
        A list of attributes to be dropped from the children. Defaults to ['score'].

    Returns
    -------
    tuple : Tuple[base.Toolbox, base.Toolbox]
        A tuple containing the two child individuals.

    """
    if drop_attrs is None:
        drop_attrs = ['score']

    # Check random_state
    rstate = check_random_state(random_state)

    # Create two empty masks for the two individuals
    mask1, mask2 = [], []

    # Iterate over the two individuals and create masks for them by swapping their attributes
    for x, y in zip(ind1.mask, ind2.mask):
        if rstate.rand() < indpb:
            mask1.append(x)
            mask2.append(y)
        else:
            mask1.append(y)
            mask2.append(x)

    # Create the two child individuals using the masks
    child1 = ind1.copy().set_mask(mask1)
    child2 = ind2.copy().set_mask(mask2)

    # Create the two child individuals using the masks
    child1.parents = (ind1, ind2)
    child2.parents = (ind1, ind2)

    # Drop specified attributes from the children
    for attr in drop_attrs:
        for child in [child1, child2]:
            if hasattr(child, attr):
                delattr(child, attr)

    # Return the two child individuals
    return child1, child2


def cxOnePoint(ind1,
               ind2,
               random_state: Optional[int] = None,
               drop_attrs: Optional[List[int]] = None) -> Tuple:

    if drop_attrs is None:
        drop_attrs = ['score']
    rstate = check_random_state(random_state)

    # Get the number of features in the individuals
    n = ind1.n_features

    # Shuffle the indices of the features
    argsort = rstate.permutation(n)

    # Choose a random crossover point
    a = rstate.randint(n)

    # Create empty masks for the two individuals
    mask1 = np.zeros((n,), dtype=bool)
    mask2 = np.zeros((n,), dtype=bool)

    # Iterate over the features and create masks for the two individuals by swapping their attributes
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

    # Create the two child individuals using the masks
    child1 = ind1.copy().set_mask(mask1)
    child2 = ind2.copy().set_mask(mask2)

    # Set the parents of the two child individuals
    child1.parents = (ind1, ind2)
    child2.parents = (ind1, ind2)

    # Drop specified attributes from the children
    for attr in drop_attrs:
        for child in [child1, child2]:
            if hasattr(child, attr):
                delattr(child, attr)

    # Return the two child individuals
    return child1, child2


def cxTwoPoint(ind1: base.Toolbox,
               ind2: base.Toolbox,
               random_state: Optional[int] = None,
               drop_attrs: Optional[str] = None) -> Tuple[base.Toolbox, base.Toolbox]:

    # If `drop_attrs` is not provided, set it to ['score'] by default
    if drop_attrs is None:
        drop_attrs = ['score']
    rstate = check_random_state(random_state)

    # Get the number of features in the individual
    n = ind1.n_features

    # Create a random permutation of the feature indices
    argsort = rstate.permutation(n)

    # Choose two distinct points a and b in the range [0, n)
    a = rstate.randint(n)
    b = rstate.randint(n)
    a, b = min(a, b), max(a, b)

    # Create two boolean masks of length n, initialized to False
    mask1 = np.zeros((n,), dtype=bool)
    mask2 = np.zeros((n,), dtype=bool)

    # Iterate over the feature indices in the permutation
    for i in range(n):
        # Get the feature index in the original order
        j = argsort[i]

        # Get the values of the corresponding features in both parents
        x = ind1.mask[i]
        y = ind2.mask[j]

        # If the feature index is between a and b, use the value from parent 1
        # Otherwise, use the value from parent 2
        if a <= i <= b:
            mask1[j] = x
            mask2[j] = y
        else:
            mask1[j] = y
            mask2[j] = x

    # Create two new individuals by copying the original individuals and setting their masks
    child1 = ind1.copy().set_mask(mask1)
    child2 = ind2.copy().set_mask(mask2)

    # Set the parents of the new individuals to be the original parents
    child1.parents = (ind1, ind2)
    child2.parents = (ind1, ind2)

    # Delete the specified attributes from the new individuals
    for attr in drop_attrs:
        for child in [child1, child2]:
            if hasattr(child, attr):
                delattr(child, attr)

    # Return the new individuals as a tuple
    return child1, child2


CROSSOVER = {
    'one': cxOnePoint,
    'two': cxTwoPoint,
    'uni': cxUniform,
}


def mutSubset(ind: base.Toolbox,
              indpb: float,
              random_state: Optional[int] = None,
              drop_attrs: List[str] = None) -> base.Toolbox:
    """
    Mutate an individual by randomly flipping the values of a subset of its features.

    Parameters
    ----------
    individual : Individual
        The individual to be mutated.
    indpb : float
        The probability of each feature to be mutated.
    random_state : int, optional
        Seed for the random number generator.
    drop_attrs : list, optional
        List of attributes to be dropped from the mutated individual.

    Returns
    -------
    Individual:
        The mutated individual.
    """

    # If no attributes to drop are specified, we drop the 'score' attribute by default
    if drop_attrs is None:
        drop_attrs = ['score']

    # Check if a random state was passed and create a random generator object accordingly
    rstate = check_random_state(random_state)

    # Create a mask for the mutation by flipping each feature in the individual with probability indpb
    mask = []
    for x in ind.mask:
        y = (rstate.rand() < indpb)
        mask.append(x ^ y)

    # Create a new individual with the mutated mask
    mutant = ind.set_mask(mask)

    # Remove the attributes to be dropped from the mutated individual
    for attr in drop_attrs:
        if hasattr(mutant, attr):
            delattr(mutant, attr)

    return mutant


class GeneticSelector(_WrappedSelector):
    """
    Feature Selector based on Differential Evolution algorithm

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

    """

    def __init__(self,
                 estimator: BaseEstimator,
                 cv: int = 5,
                 scoring: Optional[Union[str, Callable]] = None,
                 n_gen: Optional[int] = None,
                 crossover: str = 'one',
                 min_features: Union[float, int] = 0.1,
                 max_features: Union[float, int] = 0.9,
                 pop_size: int = 50,
                 mutation: float = 0.01,
                 max_time: Optional[int] = None,
                 random_state: Optional[int] = None,
                 n_jobs: Optional[int] = None,
                 verbose: int = 1,
                 n_digits: int = 4,
                 cv_kwargs: Optional[dict] = None):
        if cv_kwargs is None:
            cv_kwargs = {}
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.cv_kwargs = cv_kwargs

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

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            groups: Optional[pd.Series] = None) -> 'GeneticSelector':
        """
        Fits the genetic selector to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        y : pd.Series
            Target variable.

        groups : pd.Series, default=None
            Group labels for grouping in stratified sampling.

        Returns
        -------
        self : GeneticSelector
            Fitted genetic selector.
        """

        # Prepare data and set up selector
        self._fit_start(X)

        # Fit selector to data
        self._fit(X, y, groups)

        return self

    def partial_fit(self,
                    X: pd.DataFrame,
                    y: pd.Series,
                    groups: Optional[pd.Series] = None) -> 'GeneticSelector':
        """
        Fits the genetic selector to a partial amount of data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        y : pd.Series
            Target variable.

        groups : pd.Series, default=None
            Group labels for grouping in stratified sampling.

        Returns
        -------
        self : GeneticSelector
            Partially fitted genetic selector.
        """

        # Prepare data and set up selector
        self._fit_start(X, partial=True)

        # Partially fit selector to data
        self._fit(X, y, groups)

        return self

    def _fit_start(self,
                   X: pd.DataFrame,
                   partial: bool = False) -> 'GeneticSelector':
        """
        Initializes the genetic selector before fitting.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        partial : bool, default=False
            Whether the fitting is partial or full.

        Returns
        -------
        self : GeneticSelector
            Initialized genetic selector.
        """

        # Set features
        self._set_features(X)

        if not partial or not hasattr(self, 'trials_'):
            # Reset trials and generation count
            self._reset_trials()
            self.k_gen_ = 0

            # Initialize toolbox
            self.toolbox = base.Toolbox()
            self.rstate = check_random_state(self.random_state)

            # Define individual
            k_min = self.min_features_
            k_max = self.max_features_

            def get_individual():
                ind_size = self.rstate.choice(range(k_min, k_max + 1))
                features = self.features_.sample(ind_size)
                return features

            self.toolbox.register("individual", get_individual)

            # Define population
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            self.population = self.toolbox.population(n=self.pop_size)

        return self

    def _fit(self,
             X: pd.DataFrame,
             y: pd.Series,
             groups: pd.Series) -> 'GeneticSelector':
        """
        Fits the GeneticSelector Feature Selector model to the provided data.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature dataset.
        y : pd.Series
            The target variable.
        groups : pd.Series
            The groups variable, used for cross-validation.

        Returns
        -------
        GeneticSelector
            The fitted Genetic Feature Selector object.
        """

        # Define crossover & mutation
        mate = CROSSOVER[self.crossover]
        self.toolbox.register("mate", mate, random_state=self.rstate)
        self.toolbox.register("mutate", mutSubset, random_state=self.rstate, indpb=self.mutation)

        # Define evaluation & selection
        self.toolbox.register("eval", self.eval_subset, X=X, y=y, groups=groups)
        self.toolbox.register("select", tools.selTournament, tournsize=5, fit_attr='score')

        while not self.n_gen or self.k_gen_ < self.n_gen:

            if self.verbose:
                logmsg(f'GENERATION {self.k_gen_ + 1}')

            try:
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
                std = int(np.std(sizes))

                logmsg('SIZE AVG: {} ± {}'.format(avg, std))
                logmsg('SIZE MIN: {}'.format(np.min(sizes)))
                logmsg('SIZE MAX: {}'.format(np.max(sizes)))
                print()

                times = [ind.eval_time for ind in offspring]
                time_avg = secfmt(np.mean(times))
                time_sum = secfmt(np.sum(times))

                logmsg('TIME SUM: {}'.format(time_sum))
                logmsg('TIME AVG: {}'.format(time_avg))
                print()

        return self

    def get_subset(self) -> pd.Index:
        """
        Get the best subset of features from the last run of the genetic selector.

        Returns
        -------
        best_subset_ : pd.Index
            The best subset of features.
        """
        return self.best_subset_

    def plot_progress(self,
                      **kwargs) -> tuple:
        """
        Plot the progress of the genetic selector during fitting.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments to be passed to the plot function.

        Returns
        -------
        tuple
        """
        kwargs_ = dict(marker='.', linestyle='--', alpha=0.3, c='g')
        kwargs_.update(kwargs)
        return _plot_progress(self, **kwargs_)

    def plot_subset(self,
                    **kwargs) -> tuple:
        """
        Plot the best subset of features from the last run of the genetic selector.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments to be passed to the plot function.

        Returns
        -------
        tuple
        """
        kwargs_ = dict(marker='.', linestyle='--', alpha=0.3, c='g')
        kwargs_.update(kwargs)
        return _plot_subset(self, **kwargs_)


class GroupGeneticSelector(_WrappedGroupSelector, GeneticSelector):
    pass
