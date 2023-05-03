from functools import reduce

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from . import BaseOptimizer


class GridSearchCV(BaseOptimizer):
    """
    Exhaustive search over specified parameter values for an estimator.
    This is a subclass of BaseOptimizer, which defines the common interface
    for all hyperparameter optimization algorithms.

    Parameters:
    -----------
    estimator : object
        A Scikit-learn estimator object implementing the "fit" and "score" methods.
    base_space : dict
        A dictionary where keys are parameter names and values are tuples with
        (min_value, max_value, step) for quantitative parameters, a list of values
        for categorical parameters and a constant value for parameters that should not
        be optimized.
    btypes : dict
        A dictionary where keys are parameter names and values are strings specifying the
        type of parameter: 'quniform' for quantitative parameters with uniform distribution,
        'quniform_int' for quantitative parameters with uniform integer distribution,
        'choice' for categorical parameters and 'const' for parameters with constant value.

    Attributes:
    -----------
    max_iter : int
        The maximum number of parameter combinations that will be evaluated during the search.
    grid : ParameterGrid
        A ParameterGrid object containing all the parameter combinations that will be evaluated.
    """

    def _get_space(self, base_space: dict) -> dict:
        """
        Convert the base_space dictionary into a dictionary of possible parameter values.

        Parameters
        ----------
        base_space : dict
            A dictionary where keys are parameter names and values are tuples with
            (min_value, max_value, step) for quantitative parameters, a list of values
            for categorical parameters and a constant value for parameters that should not
            be optimized.

        Returns
        --------
        space : dict
            A dictionary where keys are parameter names and values are arrays (for quantitative
            parameters), lists (for categorical parameters) or sets (for constant parameters)
            containing all the possible values for that parameter.
        """

        # Initialize an empty dictionary
        space = {}

        # Iterate through each parameter and its type in btypes
        for param, btype in self.btypes.items():
            # If the parameter is quantitative:
            if btype in ["quniform", "quniform_int"]:
                # Extract the minimum value, maximum value and step size from base_space dictionary for the parameter
                a, b, q = base_space[param]

                # Populate the space dictionary with all the possible values for this parameter
                space[param] = np.arange(a, b + q, q)

            # If the parameter is choice:
            elif btype is "choice":
                # Populate the space dictionary with all the possible values for this parameter
                space[param] = base_space[param]

            # If the parameter is constant:
            elif btype is "const":
                # Populate the space dictionary with a set containing the constant value for this parameter
                space[param] = {base_space[param]}

            # If the parameter type is not valid, raise an error.
            else:
                raise ValueError(
                    "Continuous type of parameter bounds are not "
                    "allowed for GridSearchCV"
                )

        # Calculate the maximum number of iterations required to cover all possible parameter values in space
        self.max_iter = reduce(lambda N, grid: N * len(grid), space.values(), 1)

        # Return the space dictionary
        return space

    def _fit(
        self, X: pd.DataFrame, y: pd.Series, groups: pd.Series = None
    ) -> "GridSearchCV":
        """
        Generate all possible parameter combinations and evaluate them using the eval_params method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into train/test set.

        Returns
        -------
        self : GridSearchCV
            Returns the instance itself.
        """
        self.grid = ParameterGrid(self.space)

        for params in self.grid:
            self.eval_params(params, X, y, groups)

        return self
