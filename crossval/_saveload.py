import pandas as pd
import numpy as np

import joblib
import os
import regex


__all__ = ['save_result', 'load_result', 'remove_result']




def save_result(result, name, idx=None, detach_preds=True, path='./output',
                force_rewrite=False):

    result = dict(result)

    if idx is None:
        # If not specified, use last + 1
        idx = last_result_idx(path) + 1

    for key in ['new_pred', 'oof_pred']:
        if detach_preds and key in result:

            # Detach predictions & delete from result
            y = result[key]
            del result[key]

            # Save predictions in separate files
            prefix = key.split('_')[0] # "new" or "oof" prefix
            fpath = os.path.join(path, '{} {} {}.csv'.format(idx, prefix, name))

            y.to_csv(fpath, header=True)

            # Logging
            print('{}  ({})'.format(fpath, sizeof_format(y)))

    # Save main result
    fpath = os.path.join(path, '{} res {}.pkl'.format(idx, name))
    _ = joblib.dump(result, fpath)

    # Logging
    print('{}  ({})'.format(fpath, sizeof_format(result)))



def load_result(idx, path='./output'):

    # Load main result
    for fname in os.listdir(path):
        if regex.match('{} res .*.pkl'.format(idx), fname) is not None:
            fpath = os.path.join(path, fname)
            result = joblib.load(fpath)
            break

    # Load predictions
    for fname in os.listdir(path):
        for prefix in ['new', 'oof']:
            if regex.match('{} {} .*.csv'.format(idx, prefix), fname) is not None:
                fpath = os.path.join(path, fname)
                result['{}_pred'.format(prefix)] = pd.read_csv(fpath, index_col=0)

    return result



def remove_result(idx, path='./output'):

    for fname in os.listdir(path):

        if regex.match('{} res .*.pkl'.format(idx), fname) is not None:
            fpath = os.path.join(path, fname)
            os.remove(fpath)

        elif regex.match('{} new .*.csv'.format(idx), fname) is not None:
            fpath = os.path.join(path, fname)
            os.remove(fpath)

        elif regex.match('{} oof .*.csv'.format(idx), fname) is not None:
            fpath = os.path.join(path, fname)
            os.remove(fpath)



def last_result_idx(path='./output'):
    fnames = os.listdir(path)
    str_indices = [fname.split(' ')[0] for fname in fnames]
    int_indices = [int(idx) for idx in str_indices if idx.isdigit()]
    last_idx = max(int_indices) if len(int_indices) > 0 else 0
    return last_idx
