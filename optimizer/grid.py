import numpy as np

from sklearn.model_selection import ParameterGrid
from functools import reduce

from .base import BaseOptimizer



class GridSearchCV(BaseOptimizer):


    def _get_space(self, base_space):

        space = {}

        for param, btype in self.btypes.items():

            if btype in ['quniform', 'quniform_int']:
                a, b, q = base_space[param]
                space[param] = np.arange(a, b+q, q)

            elif btype is 'choice':
                space[param] = base_space[param]

            elif btype is 'const':
                space[param] = {base_space[param]}

            else:
                raise ValueError("Continuous type of parameter bounds are not "
                                 "allowed for GridSearchCV")

        self.max_iter = reduce(lambda N, grid: N * len(grid), space.values(), 1)
        return space


    def _fit(self, X, y, groups=None):

        self.grid = ParameterGrid(self.space)

        try:
            for params in self.grid:
                self.eval_params(params, X, y, groups)

        except KeyboardInterrupt:
            pass

        return self
