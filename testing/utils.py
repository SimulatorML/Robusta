import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter("ignore", category=UserWarning)

from sklearn.base import clone
from .models import MODELS


__all__ = [
    # utils
    'extract_model',
    'extract_model_name',
    'get_model',
    # testing
    'all_models',
    'all_regressors',
    'all_classifiers',
    'all_clusterers',
]


def all_models(types=['regressor', 'classifier']):
    for _, row in MODELS.iterrows():
        if not types or (row.type == types) \
        or (row.type and row.type in types):
            yield row.to_dict()

def all_regressors():
    return all_models('regressor')

def all_classifiers():
    return all_models('classifier')

def all_clusterers():
    return all_models('clusterer')


def get_model(name, task='regressor', **params):
    """Get model instance by name (if model is string, otherwise return model).

    Parameters
    ----------
    model : string or estimator object
        Model's short name ('XGB', 'LGB', 'RF' & etc) or an estimator object.

    task : string, {'regressor', 'classifier', 'ranker'} (default='regressor')
        Model type. Ignored if name is not string.


    Returns
    -------
    model : estimator object

    """
    if isinstance(name, str):
        # Check model name
        if name not in MODELS:
            raise ValueError("Unknown model name: {}".format(name))

        # Check task
        tasks = set(MODELS[name].keys())
        if task not in tasks:
            raise ValueError("<task> should be in {}, not '{}''".format(tasks, task))

        # Get instance copy & set params
        estimator = clone(MODELS[name][task]())
        estimator.set_params(**params)

    else:
        # Check if scikit-learn estimator
        estimator = clone(model)

    return estimator



def extract_model(estimator):

    name = estimator.__class__.__name__

    if name is 'Pipeline':

        # Check if last step is estimator
        last_step = estimator.steps[-1][1]
        msg = "Pipeline should have <predict> method on it's last step"
        assert hasattr(last_step, 'predict'), msg

        estimator = extract_model(last_step)

    elif name is 'TransformedTargetRegressor':

        regressor = estimator.regressor
        estimator = extract_model(regressor)

    elif name in ['MultiOutputClassifier', 'MultiOutputRegressor']:

        estimator = estimator.estimator
        estimator = extract_model(estimator)

    elif name in ['ClassifierChain', 'RegressorChain', 'CalibratedClassifierCV']:

        estimator = estimator.base_estimator
        estimator = extract_model(estimator)

    return estimator



def extract_model_name(estimator, short=False):
    """Extract name of estimator instance.

    Parameters
    ----------
    estimator : estimator object
        Estimator or Pipeline

    short : bool (default=False)
        Whether to return estimator's short name. For example:

            - 'XGBRegressor' -> 'XGB'
            - 'LogisticRegression' -> 'LogReg'
            - 'RandomForestClassifier' -> 'RF'
            - 'PassiveAggressiveRegressor' -> 'PA'


    Returns
    -------
    name : string
        Name of the estimator's core model

    """
    model = extract_model(estimator)
    return model.__class__.__name__
