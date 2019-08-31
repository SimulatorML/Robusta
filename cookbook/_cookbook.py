import numpy as np
import pandas as pd

from pympler.asizeof import asizeof

from ..preprocessing.category import *
from ..preprocessing.numeric import *
from ..pipeline import *


__all__ = [
    'memory_reduce_pipeline',
    'preprocessing_pipeline',
    'bytefmt',
    'sizeof',
]




memory_reduce_pipeline = FeatureUnion([
    ('numeric', make_pipeline(
        TypeSelector(np.number),
        NumericDowncast(),
    )),
    ('category', make_pipeline(
        TypeSelector('object'),
    )),
])



preprocessing_pipeline = make_pipeline(
    FeatureUnion([
        ("numeric", make_pipeline(
            TypeSelector(np.number),
            Imputer(strategy="median"),
            GaussRank(),
            ColumnRenamer(prefix='gr_'),
        )),
        ("category", make_pipeline(
            TypeSelector("object"),
            LabelEncoder(),
            ColumnRenamer(prefix='le_'),
        )),
    ])
)



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
