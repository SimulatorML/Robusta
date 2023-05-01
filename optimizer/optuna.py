from typing import Dict, Any, Union, Optional

import numpy as np
import optuna
import pandas as pd

from . import BaseOptimizer, qround


class OptunaCV(BaseOptimizer):
    """
    An optimizer that uses Optuna to search for hyperparameters.
    The search space is defined by the `space` attribute of the `BaseOptimizer` superclass.
    The objective function is defined by the `eval_params` method of the `BaseOptimizer` superclass.

    Parameters:
    -----------
    BaseOptimizer : class
        A superclass that defines the search space and the objective function.

    Attributes:
    -----------
    study : optuna.Study
        An Optuna study object that contains the hyperparameters found by the search.
    """

    @staticmethod
    def _get_space(base_space: Dict[str, Any]) -> Dict[str, Union[tuple, list]]:
        """
        Returns the search space as defined by the superclass.

        Parameters
        ----------
        base_space : dict
            The search space as defined by the superclass.

        Returns
        -------
        space : dict
            The search space as defined by the superclass.
        """
        return base_space

    def _get_params(self,
                    trial: optuna.Trial) -> Dict[str, Any]:
        """
        Returns a dictionary with hyperparameters sampled from the search space.

        Parameters
        ----------
        trial : optuna.Trial
            An Optuna trial object that is used to sample hyperparameters from the search space.

        Returns
        -------
        params : dict
            A dictionary with hyperparameters sampled from the search space.
        """
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

    def _fit(self,
             X: pd.DataFrame,
             y: pd.Series,
             groups: np.array = None) -> 'OptunaCV':
        """
        Runs the hyperparameter search using Optuna.
        Returns self.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : pd.Series
            The target values.
        groups : np.ndarray or None, optional (default=None)
            The groups for cross-validation.

        Returns
        -------
        self :
            Opt
        """
        optuna.logging.set_verbosity(optuna.logging.FATAL)
        optuna.logging.disable_default_handler()

        def objective(trial):
            params = self._get_params(trial)
            score = self.eval_params(params, X, y, groups)
            return score

        if not hasattr(self, 'study'):
            # TODO: set seed & other params
            sampler = optuna.samplers.TPESampler(seed=0)
            self.study = optuna.create_study(direction='maximize', sampler=sampler)

        self.study.optimize(objective)

        return self


class RandomSearchCV(OptunaCV):
    """
    Random search cross-validation using Optuna library for hyperparameter optimization.

    Parameters:
    -----------
    base_estimator: object
        A machine learning estimator that will be optimized using Optuna.
    param_distributions: dict
        Dictionary with parameters names (str) as keys and distributions or lists of parameters to try.
    n_trials: int, default=10
        The number of trials (hyperparameter settings) to try.
    scoring: str or callable, default=None
        A string (see sklearn.metrics.SCORERS) or a scorer callable object.
    n_jobs: int, default=-1
        Number of jobs to run in parallel. Set to -1 to use all available CPU cores.
    cv: int or callable, default=5
        If an integer, specifies the number of folds in a KFold cross-validation strategy.
        If a callable, this should be a cross-validation strategy that follows scikit-learn API.
    random_state: int or None, default=None
        Random seed used to initialize the random number generator.
    verbose: int, default=0
        Controls the verbosity: the higher, the more messages.
    """
    def _fit(self,
             X: pd.DataFrame,
             y: pd.Series,
             groups: Optional[pd.Series] = None) -> 'RandomSearchCV':
        """
        Fit the RandomSearchCV object to the input data using Optuna library.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.

        y : pd.Series
            The output data.

        groups : pd.Series, optional
            The groups to use when splitting the data into folds.

        Returns
        -------
        self : RandomSearchCV
            The fitted RandomSearchCV object.
        """
        optuna.logging.set_verbosity(optuna.logging.FATAL)
        optuna.logging.disable_default_handler()

        # Mute the logging from the Optuna library
        def objective(trial: optuna.Trial) -> float:
            params = self._get_params(trial)
            score = self.eval_params(params, X, y, groups)
            return score

        if not hasattr(self, 'study'):
            # TODO: set seed & other params
            # Set the random sampler as the default if no sampler is specified
            sampler = optuna.samplers.RandomSampler(seed=0)
            self.study = optuna.create_study(direction='maximize', sampler=sampler)

        # Run the optimization process
        self.study.optimize(objective)

        return self
