import numpy as np
import pandas as pd

from sklearn.model_selection import check_cv

from .saveload import load_result, list_results



def argsort_idx(idx_list):
    idx_arr = np.array([str(idx).split('.') for idx in idx_list])
    idx_arr = np.vectorize(int)(idx_arr)
    idx_arr = idx_arr[:,::-1].T
    return np.lexsort(idx_arr)



def check_cvs(results, X, y, groups=None):

    msg = 'Each cv must me the same'
    cvs = [result['cv'] for result in results]
    lens = [cv.get_n_splits(X, y, groups) for cv in cvs]
    assert min(lens) == max(lens), msg

    for i in range(1, len(cvs)):
        if cvs[0] != cvs[i]:
            assert cvs[0].get_n_splits(X, y, groups), msg
            folds0 = cvs[0].split(X, y, groups)
            folds1 = cvs[i].split(X, y, groups)
            for (trn0, oof0), (trn1, oof1) in zip(folds0, folds1):
                assert len(trn0) == len(trn1), msg
                assert len(oof0) == len(oof1), msg
                assert np.equal(trn0, trn1).all(), msg
                assert np.equal(oof0, oof1).all(), msg
    return cvs[0]



def load_results(idx_list=None, y_train=None,
                 result_path='./results/',
                 raise_error=False):

    if idx_list:
        idx_list = np.array(idx_list)
    else:
        fnames = list_results(result_path)
        idx_list = np.array([x.split()[0][1:-1] for x in fnames])

    arg_sort = argsort_idx(idx_list)
    idx_list = np.array(idx_list)[arg_sort]

    results = {}

    for idx in idx_list:
        try:
            result = load_result(idx, result_path)
            results[idx] = result
        except:
            if raise_error:
                raise IOError('Error while loading model #{}'.format(idx))

    return results




def split_cv_groups(results, y_train):

    hash_idx = {}
    for idx, result in results.items():
        cv = check_cv(result['cv'], y_train)

        folds = cv.split(y_train, y_train)
        h = hash(str(list(folds)))

        if h in hash_idx:
            hash_idx[h].append(idx)
        else:
            hash_idx[h] = [idx]

    cv_groups = list(hash_idx.values())
    return cv_groups



def check_folds(results, y_train):

    cv_groups = split_cv_groups(results, y_train)
    n_schemes = len(cv_groups)

    idx_abc = dict(zip(np.arange(n_schemes), cv_groups))

    if n_schemes > 1:
        print("Found {} cross-validation schemes:\n".format(n_schemes))
    else:
        print("Found SINGLE cross-validation scheme:\n")
    for i, idx in idx_abc.items():
        print('{}: {}'.format(i, idx))

    return idx_abc



def rating_table(results, n_digits=4, fold_scores=False):

    idx_list = results.keys()

    names = [result['model_name'] for result in results.values()]
    names = pd.Series(names, index=idx_list, name='MODEL_NAME')

    get_value = lambda dic, key: dic[key] if key in dic else None
    pub_score = [get_value(result, 'public_score') for result in results.values()]
    pub_score = pd.Series(pub_score, index=idx_list, name='PUBLIC')
    prv_score = [get_value(result, 'private_score') for result in results.values()]
    prv_score = pd.Series(prv_score, index=idx_list, name='PRIVATE')

    val_scores = [result['val_score'] for result in results.values()]
    df = pd.DataFrame(val_scores, index=idx_list)
    folds_cols = df.columns.map(lambda x: 'FOLD_{}'.format(x))
    df.columns = folds_cols

    df_stats = df.agg(['mean', 'std', 'min', 'max'], axis=1)
    df_stats.columns = ['LOCAL', 'STD', 'MIN', 'MAX']

    df = pd.concat([names, prv_score, pub_score, df_stats, df], axis=1)
    df.columns = df.columns.map(lambda x: x.upper())

    if not fold_scores:
        df = df.drop(columns=folds_cols)

    df = df.round(n_digits)
    return df
