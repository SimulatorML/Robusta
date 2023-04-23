from itertools import combinations, chain
from typing import List, Tuple, Union, Dict

import numpy as np
from pympler.asizeof import asizeof


def all_subsets(cols: List,
                k_range: Tuple[int, int]) -> chain:
    """
    Return an iterator that yields all possible combinations of the columns
    in `cols` with length `k` for all `k` in the range of `k_range`.

    Args:
        cols (List): a list of items to create combinations from
        k_range (Tuple[int, int]): a tuple of two integers representing the
            inclusive lower and upper bounds for the length of the subsets to
            be generated

    Returns:
        An iterator that yields all possible combinations of the columns in
        `cols` with length `k` for all `k` in the range of `k_range`.
    """
    return chain(*map(lambda k: combinations(cols, k), k_range))


def get_ranks(arr: Union[List[float], np.ndarray],
              normalize: bool = False) -> np.ndarray:
    """
    Computes the ranks of an array of numeric values.

    Args:
        arr (Union[List[float], np.ndarray]): The array of numeric values.
        normalize (bool): If True, the ranks will be normalized so that they sum to 1.
            Defaults to False.

    Returns:
        The ranks of the array, either as a numpy array of floats or as a normalized numpy array of floats.
    """
    # Convert the input array to a numpy array for convenience.
    arr = np.array(arr)

    # Get the indices of the elements in the sorted array.
    ind = np.argsort(arr)

    # Assign the ranks to the elements of the array based on their indices in the sorted array.
    arr[ind] = np.arange(len(arr))

    # Normalize the ranks if necessary.
    return arr / sum(arr) if normalize else arr


def bytefmt(n: int,
            rnd: bool = True) -> str:
    """
    Converts a number of bytes to a formatted string.

    Parameters
    ----------
    n : int
        Number of bytes to be converted.

    rnd : bool, default=True
        If True, the number of bytes will be rounded to the largest unit, e.g. '9 KB  784 bytes' -> '9.8 KB'.
        If False, the number of bytes will be represented exactly in the smallest unit.
        Ignored if <fmt> set to False.

    Returns
    -------
    str
        Formatted string representing the number of bytes.

    """
    # Define the byte units and their corresponding bases.
    byte_units = {
        'TB': 2 ** 40,
        'GB': 2 ** 30,
        'MB': 2 ** 20,
        'KB': 2 ** 10,
        'bytes': 2 ** 0,
    }

    arr = []

    # Iterate over the byte units from largest to smallest.
    for unit, base in byte_units.items():

        # Check if the number of bytes is larger than the current unit.
        if n // base > 0:

            # Compute the number of units of the current size.
            k = n // base
            n = n % base

            # If rounding is enabled and there are still bytes left, round up to the nearest unit.
            if rnd and n > 0:
                k += round(n / base, 1)
                n = 0

            # Convert to integer if possible.
            if k == int(k):
                k = int(k)

            # Append the formatted string to the array.
            arr.append('{} {}'.format(k, unit))

    # Join the array into a single string with double spaces as separator.
    return '  '.join(arr)


def sizeof(obj: object,
           fmt: bool = True,
           rnd: bool = True) -> Union[int, str]:
    """
    Return the full size of an (nested) object in bytes or a formatted string.

    Parameters
    ----------
    obj : object
        The object whose size should be determined.

    fmt : bool, default=True
        If True, return the size as a formatted string (e.g. '9.8 KB').
        Otherwise, return the size in bytes as an integer.

    rnd : bool, default=True
        If True and <fmt> is True, round the size to the largest unit.
        E.g. '9 KB  784 bytes' -> '9.8 KB'.
        Ignored if <fmt> is False.

    Returns
    -------
    Union[int, str]
        The size of the object, either as an integer number of bytes or a formatted string.

    """
    # Get the size of the object in bytes.
    n = asizeof(obj)

    # Format the size as a string if requested.
    if fmt:
        s = bytefmt(n, rnd)
    else:
        s = n

    return s


def ld2dl(ld: List[Dict]) -> Dict[str, List]:
    """
    Convert a list of dictionaries to a dictionary of lists.

    Parameters
    ----------
    ld : List[Dict]
        A list of homogeneous dictionaries.

    Returns
    -------
    Dict[str, List]
        A dictionary of lists, where each key is a key from the input dictionaries,
        and each value is a list of values corresponding to that key across all input dictionaries.

    """
    # Create a dictionary of lists, where each key is a key from the input dictionaries
    # and each value is a list of values corresponding to that key across all input dictionaries.
    dl = {key: [d[key] for d in ld] for key in ld[0].keys()}

    return dl
