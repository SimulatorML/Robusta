import os
from typing import Dict, Any, Optional, Union, List

import joblib

_SILENT_MODES = {
    'roc_auc': lambda y: 1 - y,
}


def save_result(result: Dict[str, Any],
                idx: int,
                name: str,
                make_submit: bool = True,
                force_rewrite: bool = False,
                result_path: str = './results/',
                submit_path: str = './submits/',
                silent_mode: Optional[str] = None,
                **kwargs) -> None:
    """
    Save the model results to a file and create a submission file if necessary.

    Parameters
    ----------
    result : dict
        A dictionary containing the model's results.
    idx : int
        Index number of the model.
    name : str
        Name of the model.
    make_submit : bool, optional
        Whether to create a submission file or not. Defaults to True.
    force_rewrite : bool, optional
        Whether to overwrite existing files with the same name. Defaults to False.
    result_path : str, optional
        Path to the directory where the result file will be saved. Defaults to './results/'.
    submit_path : str, optional
        Path to the directory where the submission file will be saved. Defaults to './submits/'.
    silent_mode : str, optional
        A mode that can be applied to the predictions before saving the submission file. Defaults to None.
    **kwargs:
        Additional key-value arguments that will be added to the result dictionary.

    Returns
    -------
    Nothing:
        None
    """
    result = dict(result)
    result.update(**kwargs)

    # Check if the result file already exists
    file_name = find_result(idx, result_path)

    if file_name:
        if force_rewrite:
            print('Deleting:\n' + file_name)
            os.remove(file_name)
            print()
        else:
            raise IOError("Model {} already exists!".format(idx))

    print('Creating:')
    path = os.path.join(result_path, '[{}] {}.pkl'.format(idx, name))
    _ = joblib.dump(result, path)
    print(path)

    # Create the submission file if necessary
    if make_submit and ('new_pred' in result):
        path = os.path.join(submit_path, '[{}] {}.csv'.format(idx, name))
        pred = result['new_pred']
        pred.to_csv(path, header=True)
        print(path)

        # Apply the silent mode if necessary
        if silent_mode:
            path = os.path.join(submit_path, '[{}S] {}.csv'.format(idx, name))
            pred = _SILENT_MODES[silent_mode](pred)
            pred.to_csv(path, header=True)
            print(path)
    print()


def load_result(idx: int,
                result_path: str = './results/') -> Optional[Dict[str, Union[int, str, object]]]:
    """
    Loads a result dictionary from a joblib file based on the given index and directory.

    Parameters
    ----------
    idx : int
        The index of the result to load.
    result_path : str
        The directory containing the joblib files. Default is './results/'.

    Returns
    -------
    result : dict or None
        The result dictionary loaded from the joblib file, with the 'idx' and 'model_name' keys added, or None
        if no matching file was found.
    """

    # Loop through each file in the result_path directory
    for fname in os.listdir(result_path):
        # Check if the filename starts with the given index and ends with '.pkl'
        if fname.startswith('[{}]'.format(idx)) and fname.endswith('.pkl'):
            # If the file matches the criteria, load it using joblib.load()
            path = os.path.join(result_path, fname)
            result = joblib.load(path)

            # Extract the model name from the filename
            model_name = fname.split(' ', 1)[1][:-4]

            # Add the index and model name to the result dictionary and return it
            result.update(idx=idx, model_name=model_name)
            return result
    # If no matching file was found, return None
    return None


def find_result(idx: int,
                result_path: str = './results/') -> Optional[str]:
    """
    Searches for the joblib file path containing the result with the given index.

    Parameters
    ----------
    idx : int
        The index of the result to search for.
    result_path : str
        The directory containing the joblib files. Default is './results/'.

    Returns
    -------
    path : str or None
        The file path of the joblib file containing the result with the given index, or None
        if no matching file was found.
    """

    # Loop through each file in the result_path directory
    for fname in os.listdir(result_path):
        # Check if the filename starts with the given index and ends with '.pkl'
        if fname.startswith('[{}]'.format(idx)) and fname.endswith('.pkl'):
            # If the file matches the criteria, return its path
            path = os.path.join(result_path, fname)
            return path
    # If no matching file was found, return None
    return None

def list_results(result_path: str = './results/') -> List[str]:
    """
    Returns a list of joblib file names in the given directory.

    Parameters
    ----------
    result_path : str
        The directory containing the joblib files. Default is './results/'.

    Returns
    -------
    fnames : list
        A list of joblib file names in the given directory.
    """

    # Create an empty list to store the file names
    fnames = []
    # Loop through each file in the result_path directory
    for fname in os.listdir(result_path):
        # Check if the filename ends with '.pkl'
        if fname.endswith('.pkl'):
            # If the file matches the criteria, append its name to the list
            fnames.append(fname)
    # Return the list of file names
    return fnames