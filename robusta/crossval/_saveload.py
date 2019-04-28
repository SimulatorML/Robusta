import os, shutil, regex, json, errno, warnings
from deepdiff import DeepDiff
from dictdiffer import diff
import warnings

import pandas as pd
import numpy as np
import datetime
import copy

from robusta import utils, metrics


__all__ = ['save_cv_results', 'load_cv_results', 'last_file_idx']




def last_file_idx(path='./pred'):
    fnames = os.listdir(path)
    str_indices = [fname.split(' ')[0] for fname in fnames]
    int_indices = [int(idx) for idx in str_indices if idx.isdigit()]
    last_idx = max(int_indices) if len(int_indices) > 0 else 0
    return last_idx



def save_cv_results(cv, name=None, output_path='./pred', score_digits=4):
    if name is not None:
        cv.set_name(name)
    # get results
    importance = cv.importance()
    y_sub, y_oof = cv.predict()
    cv_score, cv_std = cv.score(return_std=True)
    lb_score = None

    # folder name [de]composition
    idx = last_file_idx(output_path) + 1
    form = '%i [%.{}f] %s'.format(score_digits)
    fname = form % (idx, abs(cv_score), cv.get_name())
    path = os.path.join(output_path, fname)
    os.mkdir(path)
    csv_params = {'index': True, 'header': True}

    # datetime.txt
    dt_path = os.path.join(path, 'datetime.txt')
    with open(dt_path, 'w') as f:
        f.write(utils.dt_str(cv.datetime))

    # sub.csv, oof.csv
    sub_path = os.path.join(path, '%i sub.csv' % idx)
    oof_path = os.path.join(path, '%i oof.csv' % idx)
    if y_sub is not None: y_sub.to_csv(sub_path, **csv_params)
    if y_oof is not None: y_oof.to_csv(oof_path, **csv_params)

    # importance.csv
    imp_path = os.path.join(path, 'importance.csv')
    importance.to_csv(imp_path, **csv_params)

    # score.txt
    score_path = os.path.join(path, 'score.txt')
    with open(score_path, 'w') as f:
        f.write('metric_name: {}\n'.format(cv.metric_name))
        f.write('lb_score: {}\n'.format(lb_score))
        f.write('cv_score: {}\n'.format(cv_score))
        f.write('cv_std: {}\n'.format(cv_std))

    # model.txt
    cv_path = os.path.join(path, 'model.txt')
    with open(cv_path, 'w') as f:
        f.write('model_task: {}\n'.format(cv.model_task))
        f.write('model_name: {}\n'.format(cv.model_name))

    # cv_params.json
    ppath = os.path.join(path, 'cv_params.json')
    json.dump(cv.cv_params, open(ppath, 'w'))

    # model_params.json
    ppath = os.path.join(path, 'model_params.json')
    json.dump(cv.model_params, open(ppath, 'w'))

    # prep_params.json
    ppath = os.path.join(path, 'prep_params.json')
    json.dump(cv.prep_params, open(ppath, 'w'))

    # fit_params.json
    ppath = os.path.join(path, 'fit_params.json')
    json.dump(cv.fit_params, open(ppath, 'w'))

    # use_cols.json
    ppath = os.path.join(path, 'use_cols.json')
    json.dump(cv.use_cols, open(ppath, 'w'))

    for i in range(cv.n_folds):
        result = cv.results[i]
        fpath = os.path.join(path, 'fold{}'.format(i))
        os.mkdir(fpath)

        # importance.csv
        imp_path = os.path.join(fpath, 'importance.csv')
        result['importance'].to_csv(imp_path, **csv_params)

        # sub_pred.csv, oof_pred.csv
        sub_path = os.path.join(fpath, 'sub_pred.csv')
        oof_path = os.path.join(fpath, 'oof_pred.csv')
        if result['sub_pred'] is not None: result['sub_pred'].to_csv(sub_path, **csv_params)
        if result['oof_pred'] is not None: result['oof_pred'].to_csv(oof_path, **csv_params)

        # sub_prob.csv, oof_prob.csv
        sub_path = os.path.join(fpath, 'sub_prob.csv')
        oof_path = os.path.join(fpath, 'oof_prob.csv')
        if result['sub_prob'] is not None: result['sub_prob'].to_csv(sub_path, **csv_params)
        if result['oof_prob'] is not None: result['oof_prob'].to_csv(oof_path, **csv_params)

        # importance.csv
        imp_path = os.path.join(fpath, 'importance.csv')
        result['importance'].to_csv(imp_path, **csv_params)

        # oof_score.txt
        score_path = os.path.join(fpath, 'oof_score.txt')
        with open(score_path, 'w') as f: f.write('{}\n'.format(result['score']))

    #if len(note):
    #    note_path = os.path.join(path, 'note.txt')
    #    with open(note_path, 'w') as f: f.write('{}\n'.format(note))



def load_cv_results(cv, idx, input_path='./pred'):
    # get folder name & path

    fnames = os.listdir(input_path)
    format_test = lambda fname: regex.match('%i \[[0-9]*.[0-9]*\] .*' % idx, fname)
    search_successful = False

    for fname in fnames:
        if format_test(fname):
            search_successful = True
            break

    if search_successful == False:
        fname = '%i [score] name' % idx
        path = os.path.join(input_path, fname)
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    path = os.path.join(input_path, fname)
    cv.set_name(fname.split()[2])
    csv_params = {'index_col': 0}

    # datetime.txt
    dt_path = os.path.join(path, 'datetime.txt')
    with open(dt_path, 'r') as f:
        cv.datetime = utils.str_dt(f.readline())

    # model.txt
    cv_path = os.path.join(path, 'model.txt')
    with open(cv_path, 'r') as f:
        model_task = f.readline().split()[1]
        model_name = f.readline().split()[1]

    cv.model_task = model_task
    cv.model_name = model_name

    # score.txt
    cv_path = os.path.join(path, 'score.txt')
    with open(cv_path, 'r') as f:
        metric_name = f.readline().split()[1]

    if cv.metric_name != metric_name:
        print('Metric has changed: %s (old) -> %s (new)' % (metric_name, cv.metric_name))
        metric_update = True
    else:
        metric_update = False

    # cv_params.json
    ppath = os.path.join(path, 'cv_params.json')
    cv_params = json.load(open(ppath, 'r'))

    if cv.cv_params != cv_params:
        msg = 'Cross-Validation sheme differs:\n'
        print(cv.cv_params, cv_params)
        for key, val in DeepDiff(cv.cv_params, cv_params).items():
            msg += '{}: {}\n'.format(key, val)
        raise ValueError(msg)

    # model_params.json
    ppath = os.path.join(path, 'model_params.json')
    cv.model_params = json.load(open(ppath, 'r'))

    # prep_params.json
    ppath = os.path.join(path, 'prep_params.json')
    cv.prep_params = json.load(open(ppath, 'r'))

    # fit_params.json
    ppath = os.path.join(path, 'fit_params.json')
    cv.fit_params = json.load(open(ppath, 'r'))

    # use_cols.json
    ppath = os.path.join(path, 'use_cols.json')
    cv.use_cols = json.load(open(ppath, 'r'))

    # y_oof, y_sub
    sub_path = os.path.join(path, '%i sub.csv' % idx)
    oof_path = os.path.join(path, '%i oof.csv' % idx)
    cv.y_sub = pd.read_csv(sub_path, **csv_params)
    cv.y_oof = pd.read_csv(oof_path, **csv_params)

    # load cv-scheme
    if not hasattr(cv, 'n_folds'):
        fnames = os.listdir(path)
        fold_names = [fname for fname in fnames if regex.match(r'fold[0-9]*', fname)]
        cv.n_folds = len(fold_names)

    cv.results = []
    for i in range(cv.n_folds):
        result = {}
        fpath = os.path.join(path, 'fold{}'.format(i))

        # importance.csv
        imp_path = os.path.join(fpath, 'importance.csv')
        result['importance'] = pd.read_csv(imp_path, **csv_params)

        # sub_pred.csv, oof_pred.csv
        sub_path = os.path.join(fpath, 'sub_pred.csv')
        oof_path = os.path.join(fpath, 'oof_pred.csv')
        result['sub_pred'] = pd.read_csv(sub_path, **csv_params)
        result['oof_pred'] = pd.read_csv(oof_path, **csv_params)

        # sub_prob.csv, oof_prob.csv
        if cv.needs_proba:
            sub_path = os.path.join(fpath, 'sub_prob.csv')
            oof_path = os.path.join(fpath, 'oof_prob.csv')
            result['sub_prob'] = pd.read_csv(sub_path, **csv_params)
            result['oof_prob'] = pd.read_csv(oof_path, **csv_params)
        else:
            result['sub_prob'] = None
            result['oof_prob'] = None

        # importance.csv
        imp_path = os.path.join(fpath, 'importance.csv')
        result['importance'] = pd.read_csv(imp_path, **csv_params)

        # oof_score.txt
        score_path = os.path.join(fpath, 'oof_score.txt')
        with open(score_path, 'r') as f:
            result['score'] = float(f.readline().split()[0])

        cv.results.append(result)

    if metric_update:
        cv.update_metric(cv.metric_name)

    return cv
