#from pylab import *
import pandas as pd
import numpy as np
import warnings

from sklearn.base import BaseEstimator, clone

from ._prep import get_TT, get_TE, Imba, resamplers

from robusta import data_prep #import get_cats
from robusta import utils, metrics

from copy import copy, deepcopy


__all__ = ['Model', 'MODEL_TYPE', 'MODEL_PARAMS', 'PREP_PARAMS', 'FIT_PARAMS']



'''
Meta-model
----------
'''
class Model(BaseEstimator):
    """
    Meta-class for model.

    Parameters
    ----------
    task : string
        'reg':
            Regression task
        'bin':
            Binary Classification task

    model_name : string
        Inner estimator name.

    random_state : int, optional (default: 0)

    """

    def __init__(self, task, model_name, model_params={}):
        self.task = task
        self.model_name = model_name
        self.update(model_params)


    def fit(self, X, y=None, prep_params={}, fit_params={}):
        """Base fit function for inner estimator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Subset of the training data

        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            Subset of the target values

        cat_cols : list of strings
            Categorical columns (can include some extra columns)

        Returns
        -------
        self
        """
        X, y = self._prep(X, y, prep_params)

        fit_params = dict(fit_params)

        # check categorical features
        #for key, val in FIT_PARAMS[self.model_name]:
        #    fit_params[key]: val(X)

        # check verbose
        fit_args = utils.get_params(self.estimator.fit)
        if ('verbose' in fit_args) and ('verbose' not in fit_params):
            fit_params['verbose'] = 0

        self.estimator.fit(X, y, **fit_params)
        self.fitted_ = True
        return self


    def predict(self, X):
        """Base predict function for outer estimator.

        Parameters
        ----------
        X (array-like or sparse matrix of shape = [n_samples, n_features])
            Input features matrix.

        Returns
        -------
        self
        """
        #X = self._prep_X(X)

        y_pred = self.estimator.predict(X)
        y_pred = pd.Series(np.array(y_pred), name=self.target_name, index=X.index)
        y_pred = self._prep_y(y_pred)

        return y_pred


    def predict_proba(self, X):
        #X = self._prep_X(X)

        y_pred = self.estimator.predict_proba(X)[:,1]
        y_pred = pd.Series(np.array(y_pred), name=self.target_name, index=X.index)

        return y_pred


    def cols_importance(self, algo='inbuilt'):
        if algo in ['inbuilt', None]:
            if hasattr(self.estimator, 'feature_importances_'):
                importance = self.estimator.feature_importances_
            elif hasattr(self.estimator, 'coef_'):
                importance = abs(self.estimator.coef_)
                if len(np.shape(importance)) > 1:
                    importance = importance[0]
            else:
                importance = 1
        else:
            importance = 1

        return pd.Series(importance, index=self.cols, name='importance')


    def copy(self):
        _self = deepcopy(self)

        _self.task = self.task
        _self.model_name = self.model_name
        _self.update(self.get_params())

        return _self


    def update(self, model_params={}, init_model=True):

        if init_model:
            self._init_estimator()
            self.model_params = self.get_params()

        self.set_params(**model_params)


    def _init_estimator(self):
        output = 'There is no such model in pipeline ("%s").' % self.model_name
        assert self.model_name in MODELS, output
        undefined_model = MODELS[self.model_name]

        output = '%s is not suitable for this task ("%s").' % (self.model_name, self.task)
        assert self.task in undefined_model, output
        self.estimator = undefined_model[self.task]()


    def get_params(self, deep=True):
        model_params = self.estimator.get_params(deep=deep)
        return model_params


    def set_params(self, **model_params):
        self.estimator.set_params(**model_params)
        return self
        #model_params = {key: model_params[key] for key in model_params if key in self.get_params()}
        #self.model_params = self.get_params()


    def _prep(self, X, y, prep_params={}):

        X, y = X.copy(), y.copy()
        self.target_name = y.name
        self.cols = list(X.columns)

        params = {
            'tt_name': None,
            'te_name': 'nested',
            'te_params': {'folds': 5, 'alpha': 30},
            'imba_name': None,
            'imba_params': {'random_state': 0},
        }
        params.update(prep_params)

        tt = get_TT(params['tt_name'])
        te = get_TE(params['te_name'], params['te_params'])
        imba = Imba(params['imba_name'], **params['imba_params'])

        #cats = data_prep.get_cats(X)
        #X = data_prep.cat_le(X, cats, '')

        y = tt.transform(y) # Transform
        #X, y = imba.fit_resample(X, y) # Resample
        #X = te.fit_transform(X, y, cats) # Encode

        self.transformers = {'tt': tt, 'te': te}
        return X, y


    def _prep_X(self, X):
        return self.transformers['te'].transform(X)


    def _prep_y(self, y):
        return self.transformers['tt'].inverse_transform(y)





'''
Core Estimators
+ Basic Model's categories: Linear, Tree Based, Distance Based & etc.
+ Model's hyperparameter space
'''
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.ensemble import *
from sklearn.svm import *
from rgf import *

from bartpy.sklearnmodel import SklearnModel as BART
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from xlearn import FMModel, FFMModel
from skrvm import RVR, RVC
from pyearth import Earth

from ._blend import Blend
from ._nng import NonNegativeGarrote



MODELS = {
    'Blend': {
        'reg': Blend,
        'bin': Blend
    },
    'OLS': {
        'reg': LinearRegression
    },
    'NNG': {
        'reg': NonNegativeGarrote
    },
    'Logit': {
        'bin': LogisticRegression
    },
    'GLM': {
        'reg': SGDRegressor,
        'bin': SGDClassifier
    },
    'LinearSVM': {
        'reg': LinearSVR,
        'bin': LinearSVC
    },
    'PA': {
        'reg': PassiveAggressiveRegressor,
        'bin': PassiveAggressiveClassifier
    },
    'Ridge': {
        'reg': Ridge,
        'bin': RidgeClassifier
    },
    'BayesRidge': {
        'reg': BayesianRidge
    },
    'MARS': {
        'reg': Earth
    },
    'ARD': {
        'reg': ARDRegression
    },
    'OMP': {
        'reg': OrthogonalMatchingPursuit
    },
    'RANSAC': {
        'reg': RANSACRegressor
    },
    'TheilSen': {
        'reg': TheilSenRegressor
    },
    'Huber': {
        'reg': HuberRegressor
    },
    'Lasso': {
        'reg': Lasso
    },
    'Lars': {
        'reg': Lars
    },
    'LassoLars': {
        'reg': LassoLars
    },
    'ElasticNet': {
        'reg': ElasticNet
    },
    'PLS': {
        'reg': PLSRegression
    },


    # Tree Based
    'XGBoost': {
        'reg': XGBRegressor,
        'bin': XGBClassifier
    },
    'LightGBM': {
        'reg': LGBMRegressor,
        'bin': LGBMClassifier
    },
    'CatBoost': {
        'reg': CatBoostRegressor,
        'bin': CatBoostClassifier
    },
    'GBM': {
        'reg': GradientBoostingRegressor,
        'bin': GradientBoostingClassifier
    },
    'AdaBoost': {
        'reg': AdaBoostRegressor,
        'bin': AdaBoostClassifier
    },
    'RF': {
        'reg': RandomForestRegressor,
        'bin': RandomForestClassifier
    },
    'ET': {
        'reg': ExtraTreesRegressor,
        'bin': ExtraTreesClassifier
    },
    'RGF': {
        'reg': RGFRegressor,
        'bin': RGFClassifier
    },
    'FastRGF': {
    #    'reg': FastRGFRegressor,
    #    'bin': FastRGFClassifier
    },
    'BART': {
        'reg': BART
    },

    # Distance Based
    'KNN': {
        'reg': KNeighborsRegressor,
        'bin': KNeighborsClassifier
    },
    'RadiusNN': {
    #    'reg': RadiusNeighborsRegressor,
    #    'bin': RadiusNeighborsClassifier
    },
    'Centoid': {
        'bin': NearestCentroid
    },
    'SVM': {
        'reg': SVR,
        'bin': SVC
    },
    'NuSVM': {
        'reg': NuSVR,
        'bin': NuSVC
    },
    'RVM': {
        'reg': RVR,
        'bin': RVC
    },

    # Naive Bayes
    'NB': {
        'bin': GaussianNB
    },
    'BNB': {
        'bin': BernoulliNB
    },
    'MNB': {
        'bin': MultinomialNB
    },
    'CNB': {
        'bin': ComplementNB
    },

    # Factorization Machines
    'FM': {
        'reg': FMModel,
        'bin': FMModel
    },
    'FFM': {
        'reg': FFMModel,
        'bin': FFMModel
    },
}



LIN_MODELS = {'Blend', 'OLS', 'NNG', 'Logit', 'Ridge', 'BayesRidge', 'Lasso',
    'ElasticNet', 'Lars', 'LassoLars', 'PA', 'LinearSVM', 'Huber', 'MARS', 'ARD',
    'OMP', 'RANSAC', 'TheilSen', 'GLM', 'PLS'}

TREE_MODELS = {'XGBoost', 'LightGBM', 'CatBoost', 'GBM', 'AdaBoost', 'RF', 'ET',
    'RGF', 'FastRGF', 'BART'}

DIST_MODELS = {'Centoid', 'KNN', 'RadiusNN', 'SVM', 'NuSVM', 'RVM'}

PROB_MODELS = {'NB', 'BNB', 'MNB', 'CNB'}

FACT_MODELS = {'FM', 'FFM'}


MODEL_TYPE = {}
MODEL_TYPE.update({model_name: 'lin' for model_name in LIN_MODELS})
MODEL_TYPE.update({model_name: 'dist' for model_name in DIST_MODELS})
MODEL_TYPE.update({model_name: 'tree' for model_name in TREE_MODELS})
MODEL_TYPE.update({model_name: 'fact' for model_name in FACT_MODELS})
MODEL_TYPE.update({model_name: 'prob' for model_name in PROB_MODELS})




MODEL_PARAMS = {
    'XGBoost': {
    # https://xgboost.readthedocs.io/en/latest/parameter.html
        'learning_rate': 0.3,
        # Once your learning rate is fixed, do not change it.

        'n_estimators': (100, 2000, 100),

        'max_depth': (3, 12, 1),
        'max_leaves': {15, 31, 63, 127, 255, 511, 1023, 2047, 4095},

        'subsample': (0.1, 0.9, 0.05),
        'colsample_bytree': (0.1, 0.9, 0.05),
        'colsample_bylevel': (0.1, 0.9, 0.05),

        #'gamma': (1e-6, 1e6, 'log'),
        #'alpha': (1e-6, 1e6, 'log'),
        #'lambda': (1e-6, 1e6, 'log'),
    },

    'LightGBM': {
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
        'learning_rate': 0.1,
        # Once your learning rate is fixed, do not change it.

        'n_estimators': (100, 2000, 100),

        'max_depth': (3, 12, 1),
        'num_leaves': {15, 31, 63, 127, 255, 511, 1023, 2047, 4095},

        'bagging_fraction': (0.1, 0.9, 0.05),
        'feature_fraction': (0.1, 0.9, 0.05),

        #'lambda_l1': (1e-6, 1e6, 'log'),
        #'lambda_l2': (1e-6, 1e6, 'log'),
    },

    'CatBoost': {
    # https://catboost.ai/docs/concepts/parameter-tuning.html
        #'learning_rate': None,
        # By default, the learning rate is defined automatically based on the dataset properties and the number of iterations.
        # The automatically defined value should be close to the optimal one.

        'n_estimators': (100, 3000, 100),

        'depth': (3, 10, 1),
        'l2_leaf_reg': (1e-6, 1e6, 'log'),

        'bagging_temperature': (1e-6, 1e6, 'log'),
        'random_strength': (1e-6, 1e1, 'log'),
    },

    'MARS': {
    # https://github.com/scikit-learn-contrib/py-earth/blob/master/pyearth/earth.py
        #'max_terms': (1, 5, 1),
        #'max_degree': {1, 2},
        'penalty': (1e-6, 1e6, 'log'),
        #'endspan': (1, 30, 1),

        'enable_pruning': {True, False},
        'allow_missing': True,
        'allow_linear': {True, False},
    },

    'RGF': {
    # https://github.com/RGF-team/rgf/tree/master/python-package
        'algorithm': {"RGF", "RGF_Opt", "RGF_Sib"},
        'loss': {"LS", "Log", "Expo", "Abs"},

        'test_interval': {10, 50, 100, 200, 500, 1000},
        # For efficiency, it must be either multiple or divisor of 100.

        'max_leaf': {1000, 2000, 5000, 10000},
        # Appropriate values are data-dependent and vary from 1000 to 10000.

        'reg_depth': (1, 10, 1),

        'l2': {1, 0.1, 0.01},
        # Either 1, 0.1, or 0.01 often produces good results though with
        # exponential loss (loss='Expo') and logistic loss (loss='Log')
        # some data requires smaller values such as 1e-10 or 1e-20

        'sl2': {1, 0.1, 0.01},
        # By default equal to l2. On some data, l2/100 works well
    },

    'KNN': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        'n_neighbors': (1, 30, 1),
        'weights': {'uniform', 'distance'},
        'algorithm': {'ball_tree', 'kd_tree'},
        'leaf_size': (10, 100, 10),
        'p': {1, 2},
    },

    'RadiusNN': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html
        'radius': (1e-3, 1e2, 'log'),
        'weights': {'uniform', 'distance'},
        'algorithm': {'ball_tree', 'kd_tree'},
        'leaf_size': (10, 100, 10),
        'p': {1, 2},
    },

    'Centoid': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html
        'metric': {'euclidean', 'manhattan'},
    },

    'LinearSVM': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
        'dual': {True, False},
        'loss': {'hinge', 'squared_hinge'},

        'penalty': {'l1', 'l2'},

        'C': (1e-6, 1e6, 'log'),

        #'max_iter': (10000, 50000, 5000),

        'probability': True,
    },

    'SVM': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        'C': (1e-3, 1e2, 'log'),
        'gamma': (1e-3, 1e2, 'log'),

        'kernel': 'rbf',
        #'degree': {2, 3},

        #'shrinking': {True, False},

        #'probability': {True, False},
    },

    'NuSVM': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html
        'nu': (1e-6, 1),
        'gamma': (1e-3, 1e2, 'log'),

        #'kernel': {'linear', 'rbf', 'sigmoid'},
        'kernel': 'rbf',
        #'kernel': {'linear', 'poly', 'rbf', 'sigmoid'},
        #'degree': {2, 3},

        #'shrinking': {True, False},

        #'probability': {True, False},
    },

    'RVM': {
    # https://github.com/JamesRitchie/scikit-rvm
        'kernel': 'rbf',
        #'kernel': {'linear', 'poly', 'rbf'},
        #'degree': {2, 3},
        #'degree': 2,

        'alpha': (1e-6, 1, 'log'),
        'beta': (1e-6, 1, 'log'),

        'coef0': (1e-6, 1, 'log'),
        'coef1': (1e-6, 1, 'log'),
    },

    'GBM': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        'n_estimators': (100, 2000, 100),
        'learning_rate': (1e-6, 1, 'log'),

        'subsample': (0.1, 1),
        'max_depth': (3, 16, 1),

        'loss': {'deviance', 'exponential'},
        'criterion': {'mse', 'friedman_mse', 'mae'},
        'max_features': {'auto', 'sqrt', 'log2', None},
    },

    'AdaBoost': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
        'n_estimators': (100, 2000, 100),
        'learning_rate': (1e-6, 1, 'log'),

        'algorithm': {'SAMME.R', 'SAMME'},
    },

    'RF': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        'n_estimators': (100, 5000, 100),

        'min_samples_split': (1e-6, 1e-3, 'log'),
        'max_features': {'auto', 'sqrt', 'log2', None},

        'bootstrap': {True, False},
        'oob_score': True,
    },

    'ET': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
        'n_estimators': (100, 5000, 100),

        'min_samples_split': (1e-6, 1e-3, 'log'),
        'max_features': {'auto', 'sqrt', 'log2', None},

        'bootstrap': {True, False},
        'oob_score': True,
    },

    'OLS': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        'fit_intercept': {True, False},
    },

    'Logit': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        'penalty': {'l1', 'l2'},
        'C': (1e-6, 1e6, 'log'),

        'dual': {True, False},
        'fit_intercept': {True, False},

        'solver': {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
        'max_iter': 1000,
    },

    'GLM': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        'loss': {'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'},
        'fit_intercept': {True, False},

        'penalty': {'none', 'l2', 'l1', 'elasticnet'},
        'alpha': (1e-6, 1e6, 'log'),
        'l1_ratio': (0.0, 1.0),
        'epsilon': (1e-6, 1, 'log'),

        'learning_rate': {'constant', 'optimal', 'invscaling', 'adaptive'},
        'eta0': (1e-6, 1, 'log'),
        'power_t': (0, 2),
        'max_iter': (1000, 10000, 1000),
    },

    'PA': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html
        'C': (1e-6, 1e6, 'log'),
        'max_iter': (1000, 10000, 1000),

        'fit_intercept': {True, False},
        'loss': {'hinge', 'squared_hinge'},
    },

    'Ridge': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
        'alpha': (1e-6, 1e6, 'log'),

        'fit_intercept': {True, False},
        'normalize': {True, False},

        'solver': {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'},
    },

    'Lasso': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        'alpha': (1e-6, 1e6, 'log'),

        'fit_intercept': {True, False},
        'normalize': {True, False},
        'positive': {True, False},

        'selection': {'cyclic', 'random'},
    },

    'LassoLars': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html
        'alpha': (1e-6, 1e6, 'log'),
        'eps': (1e-6, 1e-2, 'log'),

        'fit_intercept': {True, False},
        'normalize': {True, False},
        'positive': {True, False},
    },

    'Lars': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html
        'eps': (1e-6, 1e-2, 'log'),

        'n_nonzero_coefs': lambda X: (1, np.shape(X)[1], 1),

        'fit_intercept': {True, False},
        'normalize': {True, False},
        'positive': {True, False},
    },

    'BART': {
    # https://github.com/JakeColtman/bartpy/blob/master/bartpy/sklearnmodel.py
        'n_trees': (100, 2000, 100),
        'n_chains': (1, 32, 1),

        'sigma_a': (1e-6, 1, 'log'),
        'sigma_b': (1e-6, 1, 'log'),

        'thin': (0, 1),
        'p_grow': (0, 1),
        'p_prune': (0, 1),

        'alpha': (0, 1),
        'beta': (0, 1e5, 'log'),

        'store_in_sample_predictions': False,
    },

    'PLS': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html
        'n_components': lambda X: (1, np.shape(X)[1], 1),
        'scale': {True, False},
    }
}





FIT_PARAMS = {
# X dependent parameters
    'Blend': {
        'weights': lambda X, use_cols: [(0,1)]*(np.shape(X)[1] if not use_cols else len(use_cols))
    },
    'CatBoost': {
        'cat_features': lambda X: get_cats(X, as_idx=True)
    },
    'LightGBM': {
        'categorical_feature': lambda X: get_cats(X, as_idx=True)
    },
}


CAT_PARAMS = {
    'CatBoost': 'cat_features',
    'LightGBM': 'categorical_feature'
}



PREP_PARAMS = {
    'tt_name': None,
    'te_name': {'nested', 'inbuilt'},
    'te_params': {
        'folds': (3, 20, 1),
        'alpha': (0, 1e3, 'log'),
    },
    'imba_name': set(resamplers.keys()),
    'imba_params': {
        'random_state': 0
    },
}
