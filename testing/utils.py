import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter("ignore", category=UserWarning)

from sklearn.base import clone
from .estimators import ESTIMATORS
from .params import PARAM_SPACE


__all__ = [
    # utils
    'get_estimator',
    'get_estimator_name',
    'extract_param_space',
    'extract_model',
    'extract_model_name',
    # testing
    'all_estimators',
    'all_regressors',
    'all_classifiers',
    'all_clusterers',
    'all_transformers',
]


def all_estimators(type_filter=['regressor', 'classifier']):
    for _, row in ESTIMATORS.iterrows():
        if not type_filter or (row.type == type_filter) \
        or (row.type and row.type in type_filter):
            yield row.to_dict()

def all_regressors():
    return all_estimators('regressor')

def all_classifiers():
    return all_estimators('classifier')

def all_clusterers():
    return all_estimators('clusterer')

def all_transformers():
    return all_estimators('transformer')


def get_estimator(estimator, estimator_type=None, **params):
    """Get model instance by name (if model is string, otherwise return model).

    Parameters
    ----------
    estimator : string or estimator object
        Model's short name ('XGB', 'LGB', 'RF' & etc) or an estimator object.

    estimator_type : {'regressor', 'classifier', ...} or None (default=None)
        Estimator type. Ignored if <estimator> is not string or if estimator
        has single type (like any transformer).

    Returns
    -------
    model : estimator object

    """

    if isinstance(estimator, str):

        name_mask = (ESTIMATORS['name'] == estimator)
        type_mask = (ESTIMATORS['type'] == estimator_type)

        # check estimator name
        if not name_mask.any():
            raise ValueError(f"Unknown estimator: '{estimator}'")

        # if has single type
        if name_mask.sum() == 1:
            estimator = ESTIMATORS[name_mask]['class'].iloc[0]()
            estimator = clone(estimator).set_params(**params)
            return estimator

        # check estimator type
        if not type_mask.any():
            raise ValueError(f"Unknown estimator type: '{estimator_type}'")

        # check if pair is in ESTIMATORS
        if not (name_mask & type_mask).any():
            raise ValueError(f"Coluld not find ('{name}', '{estimator_type}') pair")
        else:
            estimator = ESTIMATORS[name_mask & type_mask]['class'].iloc[0]()
            estimator = clone(estimator).set_params(**params)
            return estimator

    elif hassatr(name, 'fit'):
        return estimator.set_params(**params)

    else:
        raise TypeError("Unknown <estimator> type passed")


def get_estimator_name(estimator, short=False):
    name = estimator.__class__.__name__
    name_mask = (ESTIMATORS['class_name'] == name)
    if short and name_mask.any():
        return ESTIMATORS[name_mask]['name'].iloc[0]
    else:
        return name


def extract_param_space(estimator, verbose=True):

    params = estimator.get_params()
    param_names = {}
    param_space = {}

    # Find Estimators
    for key, val in params.items():

        if not hasattr(val, 'fit'):
            continue

        name = get_estimator_name(val, short=True)
        if name not in PARAM_SPACE:
            continue

        for param, space in PARAM_SPACE[name].items():
            param_space[f"{key}__{param}"] = space

        param_names[key] = val

    # Verbose
    if verbose:
        print('FOUND MODELS:')
        #display(pd.Series(param_names))
        display(param_names)
        print()

        print('FOUND PARAMETERS:')
        #display(pd.Series(param_space))
        display(param_space)
        print()

    return param_space



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
