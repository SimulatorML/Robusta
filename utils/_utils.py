import pandas as pd
import numpy as np

from pympler.asizeof import asizeof

from itertools import combinations, chain




def all_subsets(cols, k_range):
    return chain(*map(lambda k: combinations(cols, k), k_range))



def get_ranks(arr, normalize=False):
    arr = np.array(arr)
    ind = np.argsort(arr)
    arr[ind] = np.arange(len(arr))
    return arr / sum(arr) if normalize else arr



def bytefmt(n, rnd=True):
    '''Convert number of bytes to string.
    See <rnd> parameter documentation.

    Parameters
    ----------
    n : int
        Number of bytes

    rnd : bool, default=True
        If True, return number of bytes, rounded to largest unit.
        E.g. '9 KB  784 bytes' -> '9.8 KB'.
        Ignored if <fmt> set to False.


    Returns
    -------
    s : string
        Formatted string

    '''

    byte_units = {
        'TB': 2**40,
        'GB': 2**30,
        'MB': 2**20,
        'KB': 2**10,
        'bytes': 2**0,
    }

    arr = []

    for unit, base in byte_units.items():

        if n // base > 0:

            k = n // base
            n = n % base

            if rnd:
                k += round(n / base, 1)
                n = 0

            if k == int(k):
                k = int(k)

            arr.append('{} {}'.format(k, unit))

    return '  '.join(arr)



def sizeof(obj, fmt=True, rnd=True):
    '''Return full size of (nested) object.

    Parameters
    ----------
    obj : object
        Any object to determine memory size

    fmt : bool, default=True
        If True, return formatted size (string). E.g. 10000 -> '9.8 KB'.
        Otherwise return number of bytes (integer).

    rnd : bool, default=True
        If True, return number of bytes, rounded to largest unit.
        E.g. '9 KB  784 bytes' -> '9.8 KB'.
        Ignored if <fmt> set to False.


    Returns
    -------
    s : int or string
        Formatted string or exact number of bytes

    '''

    n = asizeof(obj)
    s = bytefmt(n, rnd) if fmt else n
    return s



def ld2dl(ld):
    '''Convert list of dict to dict of list

    Parameters
    ----------
    ld : list of dict
        List of homogeneous dictionaries

    Returns
    -------
    dl : dict of list
        Dictionary of list of equal length

    '''
    dl = {key: [d[key] for d in ld] for key in ld[0].keys()}
    return dl
