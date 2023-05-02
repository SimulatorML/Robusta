import warnings
from typing import Optional, List

from IPython.core.display_functions import display

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter("ignore", category=UserWarning)

from sklearn.base import BaseEstimator

from .estimators import ESTIMATORS
from .params import PARAM_SPACE


def all_estimators(type_filter: List[str] = None) -> dict:
    """
    Return a generator that yields dictionaries representing all scikit-learn
    estimators that match the given type filter. If no filter is given, all
    estimators are yielded.

    Parameters
    ----------
    type_filter : list of str, default ['regressor', 'classifier']
        A list of estimator types to include. Each type can be one of the
        following: 'regressor', 'classifier', 'transformer', or 'clusterer'.
        If an estimator has multiple types, it will be included if any of its
        types match the filter.

    Yields
    ------
    dict
        A dictionary representing a scikit-learn estimator. The dictionary
        includes the following keys: 'name', 'class', 'module', 'type'.
    """

    # Iterate over all estimators and yield them if they match the type filter
    if type_filter is None:
        type_filter = ['regressor', 'classifier']
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


def get_estimator(estimator: BaseEstimator,
                  estimator_type: Optional[str] = None,
                  **params) -> object:
    """
    Get model instance by name (if model is string, otherwise return model).

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
            estimator = ESTIMATORS[name_mask]['class'].iloc[0](**params)
            return estimator

        # check estimator type
        if not type_mask.any():
            raise ValueError(f"Unknown estimator type: '{estimator_type}'")

        # check if pair is in ESTIMATORS
        if not (name_mask & type_mask).any():
            raise ValueError(f"Coluld not find ('{name}', '{estimator_type}') pair")
        else:
            estimator = ESTIMATORS[name_mask & type_mask]['class'].iloc[0](**params)
            return estimator

    elif hasattr(name, 'fit'):
        return estimator.set_params(**params)

    else:
        raise TypeError("Unknown <estimator> type passed")


def get_estimator_name(estimator: BaseEstimator,
                       short: bool = False) -> str:
    """
    Returns the name of an estimator class.

    Parameters
    ----------
    estimator : BaseEstimator
        An instance of an estimator class.
    short : bool
        A boolean indicating whether to return a shortened version of the name if available.

    Returns
    -------
    str:
        The name of the estimator class as a string.
    """
    # Get the name of the estimator class
    name = estimator.__class__.__name__

    # Check if a shortened name is requested and available
    name_mask = (ESTIMATORS['class_name'] == name)
    if short and name_mask.any():
        # Return the shortened name if available
        return ESTIMATORS[name_mask]['name'].iloc[0]
    else:
        # Return the full name if no shortened name is available
        return name


def extract_param_space(estimator: BaseEstimator,
                        verbose: bool = True) -> dict:
    """
    Extract params

    Parameters
    ----------
    estimator : BaseEstimator
        An instance of an estimator class.
    verbose : bool
        Output state

    Returns
    -------
    dict:
        Param space
    """
    params = estimator.get_params()
    param_names = {}
    param_space = {}

    # Check if Estimator itself
    if hasattr(estimator, 'fit'):
        name = get_estimator_name(estimator, short=True)

        if name in PARAM_SPACE:
            param_names[name] = estimator
            param_space.update(PARAM_SPACE[name])

    # Find Estimators in Params
    for key, val in params.items():

        if hasattr(val, 'fit'):
            name = get_estimator_name(val, short=True)

            if name in PARAM_SPACE:
                for param, space in PARAM_SPACE[name].items():
                    param_space[f"{key}__{param}"] = space

                param_names[key] = val

    # Verbose
    if verbose:
        print('FOUND MODELS:')
        # display(pd.Series(param_names))
        display(param_names)
        print()

        print('FOUND PARAMETERS:')
        # display(pd.Series(param_space))
        display(param_space)
        print()

    return param_space


def extract_model(estimator: BaseEstimator) -> BaseEstimator:
    """
    Extract model from estimator

    Parameters
    ----------
    estimator : BaseEstimator
        An instance of an estimator class.

    Returns
    -------
    estimator:
        BaseEstimator
    """
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


def extract_model_name(estimator: BaseEstimator,
                       short: bool = False):
    """
    Extract name of estimator instance.

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
