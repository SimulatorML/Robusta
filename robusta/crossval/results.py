from typing import Optional, Any, List, Dict, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import check_cv

from .saveload import list_results
from .saveload import load_result


def argsort_idx(idx_list: np.ndarray) -> np.ndarray:
    """
    Returns the indices that would sort a list of indices in a natural order.

    Parameters
    ----------
    idx_list : np.ndarray
        a array of indices in string or integer format

    Returns
    -------
    sorted_indices : np.ndarray
        a numpy array of indices that would sort the input list
    """

    # Convert the list of indices to a numpy array of integers
    idx_arr = np.array([str(idx).split(".") for idx in idx_list])
    idx_arr = np.vectorize(int)(idx_arr)

    # Transpose the array and sort the rows using lexicographic order
    idx_arr = idx_arr[:, ::-1].T
    sorted_indices = np.lexsort(idx_arr)

    return sorted_indices


def check_cvs(
    results: List[Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series] = None,
) -> Any:
    """
    Check that all cross-validation splits are the same.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of cross-validation results, where each result is a dictionary
        containing the keys 'cv', 'params', and 'mean_score' (among others).
        The 'cv' key should contain the cross-validator used to split the data.
    X : pd.DataFrame
        The data to be split.
    y : pd.Series
        The target variable.
    groups : np.ndarray or None, optional (default=None)
        The group labels for the samples. Only used if the cross-validator
        requires group information to split the data.

    Returns
    -------
    cv0 : object
        The first cross-validator object in the `results` list.
    """
    # Check if all cross-validation splits are the same
    msg = "Each cv must be the same"
    cvs = [result["cv"] for result in results]
    lens = [cv.get_n_splits(X, y, groups) for cv in cvs]
    assert min(lens) == max(lens), msg

    # Compare each split of the first cv to all other splits of the other cvas
    cv0 = cvs[0]
    for cv_i in cvs[1:]:
        if cv0 != cv_i:
            assert cv0.get_n_splits(X, y, groups), msg
            for (trn0, oof0), (trn_i, oof_i) in zip(
                cv0.split(X, y, groups), cv_i.split(X, y, groups)
            ):
                assert len(trn0) == len(trn_i), msg
                assert len(oof0) == len(oof_i), msg
                assert np.array_equal(trn0, trn_i), msg
                assert np.array_equal(oof0, oof_i), msg

    return cv0


def load_results(
    idx_list: Optional[List[int]] = None,
    result_path: str = "./results/",
    raise_error: bool = False,
) -> Dict[Any, dict[str, int | str | object] | None]:
    """
    Load results from saved numpy arrays for a list of model indices.

    Parameters
    ----------
    idx_list : Optional[List[int]]
        A list of model indices to load. If None, all saved models in `result_path` directory will be loaded.
    result_path : str
        The directory where saved results are located.
    raise_error : bool
        Whether to raise an error if loading a result fails.

    Returns
    -------
    results : Dict[int, np.ndarray]
        A dictionary with model indices as keys and loaded numpy arrays as values.
    """

    # Check if idx_list is not None
    if idx_list:
        # Convert the input list to a numpy array
        idx_list = np.array(idx_list)

    else:
        # If idx_list is None, get the list of filenames in the result_path directory
        fnames = list_results(result_path=result_path)

        # Extract the model number from the filename and convert to a numpy array
        idx_list = np.array([x.split()[0][1:-1] for x in fnames])

    # Sort the model numbers in ascending order
    arg_sort = argsort_idx(idx_list=idx_list)

    # Reorder idx_list based on the sorted index
    idx_list = np.array(idx_list)[arg_sort]

    # Create an empty dictionary to store the results
    results = {}

    # Loop through each model number in idx_list
    for idx in idx_list:
        # Try to load the result for the given model number
        try:
            result = load_result(idx=idx, result_path=result_path)

            # If successful, add the result to the dictionary using the model number as the key
            results[idx] = result

        # If loading the result fails
        except (Exception,):
            # Check if raise_error is True, if so, raise an error
            if raise_error:
                raise IOError("Error while loading model #{}".format(idx))

    # Return the results dictionary
    return results


def split_cv_groups(results: Dict[str, Any], y_train: List[Any]) -> List[List[str]]:
    """
    Splits cross-validation (CV) results into groups with the same CV fold assignments.

    Parameters
    ----------
    results : dict
        A dictionary mapping string keys to dictionaries that contain cross-validation results. Each dictionary
        must contain a key 'cv' that maps to a scikit-learn BaseCrossValidator object. Other keys can be present
        but are not used in this function.
    y_train : list
        A list of target values for the training data.

    Returns
    -------
    list :
        A list of lists of string keys. Each inner list contains the keys of the input `results` dictionary that
        correspond to cross-validation results with the same fold assignments.

    """

    # Create an empty dictionary to store hash values as keys and list
    # of keys from 'results' dictionary as values.
    hash_idx = {}
    for idx, result in results.items():
        # Get the cross-validator object from the current result dictionary
        cv = check_cv(result["cv"], y_train)

        # Split the training data into CV folds using the current cross-validator object
        folds = cv.split(y_train, y_train)

        # Hash the fold assignments to a unique string
        h = hash(str(list(folds)))

        # If the hash value already exists in the dictionary, append the current key to its value list.
        # Otherwise, create a new key-value pair in the dictionary.
        if h in hash_idx:
            hash_idx[h].append(idx)
        else:
            hash_idx[h] = [idx]
    # Return the values from the dictionary as a list of lists
    cv_groups = list(hash_idx.values())
    return cv_groups


def check_folds(results: Dict[str, Any], y_train: List[Any]) -> Dict[int, List[int]]:
    """
    Check the cross-validation schemes used in a machine learning model.

    Parameters
    ----------
    results : Dict[str, Any]
        A dictionary containing the results of a machine learning model's cross-validation.
    y_train : List[Any]
        A list of the target values for the training data.

    Returns
    -------
    Dict[int, List[int]]:
        A dictionary mapping the index of each cross-validation scheme to a list of the indices of the training data
        used in that scheme.
    """

    # Split the cross-validation groups using the `split_cv_groups` function
    cv_groups = split_cv_groups(results=results, y_train=y_train)

    # Count the number of cross-validation groups
    n_schemes = len(cv_groups)

    # Create a dictionary of cross-validation group indices
    idx_abc = dict(zip(np.arange(n_schemes), cv_groups))

    # Print a message indicating the number of cross-validation schemes found
    if n_schemes > 1:
        print("Found {} cross-validation schemes:\n".format(n_schemes))
    else:
        print("Found SINGLE cross-validation scheme:\n")

    # Print the index of each cross-validation group
    for i, idx in idx_abc.items():
        print("{}: {}".format(i, idx))

    # Return the dictionary of cross-validation group indices
    return idx_abc


def rating_table(
    results: Dict[str, Dict[str, float]], n_digits: int = 4, fold_scores: bool = False
) -> pd.DataFrame:
    """
    Creates a table containing the performance metrics of different models on a dataset.

    Parameters
    ----------
    results:
        A dictionary containing the performance metrics for different models.
        The keys of the dictionary are the unique IDs of the models, and the values
        are dictionaries containing the following keys:
          - 'model_name': (str) The name of the model.
          - 'public_score': (float or None) The public score of the model.
          - 'private_score': (float or None) The private score of the model.
          - 'val_score': (list of float) The validation scores of the model on different folds.
    n_digits : int
        The number of digits to round the scores to.
    fold_scores: bool
        Whether to include the validation scores for different folds in the output.

    Returns
    -------
    dataframe:
        A pandas DataFrame containing the performance metrics for different models. The columns of
        the DataFrame are as follows:
            - 'MODEL_NAME': (str) The name of the model.
            - 'PUBLIC': (float or NaN) The public score of the model.
            - 'PRIVATE': (float or NaN) The private score of the model.
            - 'LOCAL': (float) The mean validation score of the model across different folds.
            - 'STD': (float) The standard deviation of the validation scores of the model across different folds.
            - 'MIN': (float) The minimum validation score of the model across different folds.
            - 'MAX': (float) The maximum validation score of the model across different folds.
    """
    # Get the IDs of the models
    idx_list = results.keys()

    # Get the names of the models
    names = [result["model_name"] for result in results.values()]
    names = pd.Series(names, index=idx_list, name="MODEL_NAME")

    # Get the public and private scores of the models
    get_value = lambda dic, key: dic[key] if key in dic else None
    pub_score = [get_value(result, "public_score") for result in results.values()]
    pub_score = pd.Series(pub_score, index=idx_list, name="PUBLIC")
    prv_score = [get_value(result, "private_score") for result in results.values()]
    prv_score = pd.Series(prv_score, index=idx_list, name="PRIVATE")

    # Get the validation scores of the models on different folds
    val_scores = [result["val_score"] for result in results.values()]
    df = pd.DataFrame(val_scores, index=idx_list)
    folds_cols = df.columns.map(lambda x: "FOLD_{}".format(x))
    df.columns = folds_cols

    # Calculate the mean, standard deviation, minimum, and maximum validation scores of the models
    df_stats = df.agg(["mean", "std", "min", "max"], axis=1)
    df_stats.columns = ["LOCAL", "STD", "MIN", "MAX"]

    # Combine the different metrics into a single DataFrame
    df = pd.concat([names, prv_score, pub_score, df_stats, df], axis=1)
    df.columns = df.columns.map(lambda x: x.upper())

    # Drop the validation scores for different folds (if fold_scores=False)
    if not fold_scores:
        df = df.drop(columns=folds_cols)

    df = df.round(n_digits)
    return df
