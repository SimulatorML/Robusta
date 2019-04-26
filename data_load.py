import pandas as pd
import numpy as np
import os, regex


def read_target(path, target, **csv_params):
    path = os.path.join(path, 'train.csv')
    header = pd.read_csv(path, nrows=1, **csv_params).columns
    index_col = header[0]
    y_train = pd.read_csv(path, index_col=index_col, usecols=[index_col, target])
    y_train = y_train[target]
    return y_train


def read_dataset(paths, target=None, cols_format=True, **csv_params):
    # first path in paths – for data with target
    if isinstance(paths, str):
        paths = [paths]
        
    X_tr_list = []
    X_ts_list = []
    for i, path in enumerate(paths):
        X_tr, X_ts = read_data(path, **csv_params)
        # y_train
        if i == 0 and target != None:
            y_tr = X_tr[target]
            X_tr.drop(columns=target, inplace=True)
        # features name
        if cols_format:
            # ('folder', 'col')
            folder = os.path.basename(os.path.normpath(path))
            X_cols = [(folder, col) for col in X_tr.columns]
            X_cols = pd.MultiIndex.from_tuples(X_cols)
        else:
            # 'col'
            X_cols = list(X_tr.columns)
        X_tr.columns = X_cols
        X_ts.columns = X_cols
        X_tr_list.append(X_tr)
        X_ts_list.append(X_ts)
            
    X_tr = pd.concat(X_tr_list, axis=1)
    X_ts = pd.concat(X_ts_list, axis=1)
    
    return X_tr, X_ts, y_tr



def read_data(input_path, **csv_params):
    
    fnames = os.listdir(input_path)
    
    for fname in fnames:
        if fname == 'train.csv':
            train_path = os.path.join(input_path, fname)
            break
        
    for fname in fnames:
        if fname == 'test.csv':
            test_path = os.path.join(input_path, fname)
            break
            
    df_train = pd.read_csv(train_path, index_col=0, **csv_params)
    df_test = pd.read_csv(test_path, index_col=0, **csv_params)
            
    return df_train, df_test