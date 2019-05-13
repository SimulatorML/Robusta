import pandas as pd
import numpy as np
import datetime
import copy

from statsmodels.stats.weightstats import ztest
from scipy.stats import ttest_ind, ttest_rel

from functools import reduce

from robusta import utils, metrics

from ._stack import *
from ._output import *
from ._saveload import *


__all__ = ['Holdout', 'KFoldCV', 'NKFoldCV', 'LOO', 'GroupLOO', 'GroupKFoldCV']




class Validator():

    def __init__(self, metric_name, verbose=2, plot=False, **cv_params):

        self.update_metric(metric_name)
        self.cv_params = cv_params

        self.verbose = verbose
        self.plot = plot

        n = 1
        if 'n_splits' in cv_params:
            n *= cv_params['n_splits']
        if 'n_repeats' in cv_params:
            n *= cv_params['n_repeats']
        self.n_folds = n



    def split(self, X_train, X_test, y_train, **data_params):
        # just trivial example
        self._split(X_train, X_test, y_train)

        ii = range(len(y_train))
        self.folds = list(zip(ii, ii))
        return self



    def _split(self, X_train, X_test, y_train=None):
        self.X_train = X_train
        self.X_test = X_test
        self.cols = list(X_train.columns)

        if y_train is not None:
            self.set_target(y_train)



    def set_target(self, y_train):
        self.target = y_train.name
        self.y_train = y_train
        return self



    def update_metric(self, metric_name):
        self.metric_name = str(metric_name)
        self.metric, self.task = metrics.get_metric(self.metric_name, return_task=True)
        self.needs_proba = metrics.needs_proba(self.metric)

        if hasattr(self, 'results'):
            scores = self._custom_scores(self.metric_name)
            for i in range(len(self.folds)):
                self.results[i]['score'] = scores[i]
        return self



    def _custom_scores(self, metric_name):

        metric = metrics.get_metric(metric_name)
        needs_proba = metrics.needs_proba(metric)

        scores = []
        for result, fold in zip(self.results, self.folds):
            trn, val = fold
            y_val = self.y_train.iloc[val]

            if self.needs_proba:
                score = metric(y_val, result['oof_prob'])
            else:
                score = metric(y_val, result['oof_pred'])
            scores.append(score)

        return scores



    def fit(self, model, model_params={}, prep_params={}, fit_params={}, use_cols=None):

        assert hasattr(self, 'X_train'), 'Fit validation to the X first.'
        assert hasattr(self, 'y_train'), 'Fit validation to the y first.'

        # save attributes
        self.model_task = model.task
        self.model_name = model.model_name

        self.model_params = model_params
        self.prep_params = prep_params
        self.fit_params = fit_params

        self.use_cols = list(use_cols) if use_cols else list(self.cols)

        tech_params = {'random_state': 0, 'verbose': 0, 'n_jobs': -1}
        tech_params = {param: tech_params[param] for param in model.get_params() if param in tech_params}

        self.results = []

        for i, fold in enumerate(self.folds):
            time_start = utils.ctime()

            trn, val = fold
            X_trn, y_trn = self.X_train.iloc[trn], self.y_train.iloc[trn]
            X_val, y_val = self.X_train.iloc[val], self.y_train.iloc[val]

            model = model.copy()
            model.set_params(**tech_params).set_params(**model_params)
            model.fit(X_trn[self.use_cols], y_trn, prep_params, fit_params)

            result = self._get_result(model, self.X_test[self.use_cols], X_val[self.use_cols], y_val)
            result['time'] = (utils.ctime() - time_start).total_seconds()

            self.results.append(result)
            self.output()

        self.datetime = datetime.datetime.now()

        return self



    def _get_result(self, model, X_tst, X_val, y_val):
        result = {}

        # importance
        result['importance'] = model.cols_importance()

        # predict
        result['oof_pred'] = model.predict(X_val)
        result['sub_pred'] = model.predict(X_tst)

        # predict_proba
        if self.needs_proba:
            result['oof_prob'] = model.predict_proba(X_val)
            result['sub_prob'] = model.predict_proba(X_tst)
            result['score'] = self.metric(y_val, result['oof_prob'])
        else:
            result['oof_prob'] = None
            result['sub_prob'] = None
            result['score'] = self.metric(y_val, result['oof_pred'])

        return result



    def fit_result(self, model, model_params={}, prep_params={}, fit_params={}, use_cols=None):

        self.fit(model, model_params, prep_params, fit_params, use_cols)
        return self.results



    def get_results(self, key='score'):

        return [result[key] for result in self.results]



    def set_results(self, results):

        cv.results = results
        return self



    def clear(self):

        for attr in ['results', 'model_name', 'name', 'model_params', 'prep_params', 'fit_params', 'use_cols']:
            if hasattr(self, attr):
                delattr(self, attr)
        return self



    def score(self, metric_name=None, return_std=False):

        if metric_name is None:
            scores = self.get_results('score')
        else:
            scores = self._custom_scores(metric_name)

        if return_std:
            return np.mean(scores), np.std(scores)
        else:
            return np.mean(scores)



    def fit_score(self, model, model_params={}, prep_params={}, fit_params={}, use_cols=None,
                  metric_name=None, return_std=False):

        self.fit(model, model_params, prep_params, fit_params, use_cols)
        return self.score(metric_name, return_std)



    def importance(self, algo=None):

        importances = self.get_results('importance')
        importance = sum(importances)/len(importances)
        importance.name = 'importance'
        return importance



    def fit_importance(self, model, model_params={}, prep_params={}, fit_params={}, use_cols=None):

        self.fit(model, model_params, prep_params, fit_params, use_cols)
        return self.importance(None)



    def time(self):
        return sum(self.get_results('time'))



    def predict(self, mean_func=None, return_oof=True):

        if self.task == 'bin':
            sub_preds = self.get_results('sub_prob')
            oof_preds = self.get_results('oof_prob')
        else:
            sub_preds = self.get_results('sub_pred')
            oof_preds = self.get_results('oof_pred')

        # init prediction series
        y_sub_base = pd.Series(0, index=self.X_test.index)
        y_oof_base = pd.Series(0, index=self.X_train.index)

        # results unpack & format
        y_subs = []
        y_oofs = []
        for sub_pred, oof_pred in zip(sub_preds, oof_preds):
            y_subs.append(y_sub_base.copy() + sub_pred)
            y_oofs.append(y_oof_base.copy() + oof_pred)

        # averaging (if n_iters > 1)
        mean_func = np.mean if mean_func is None else mean_func
        y_sub = pd.concat(y_subs, axis=1).apply(mean_func, axis=1).rename(self.target)
        y_oof = pd.concat(y_oofs, axis=1).apply(mean_func, axis=1).rename(self.target)

        # needs proba?
        if self.task == 'bin' and self.needs_proba == False:
            y_sub = (y_sub > 0.5)*1
            y_oof = (y_oof > 0.5)*1

        self.y_oof = y_oof
        self.y_sub = y_sub

        if return_oof:
            return y_sub, y_oof
        else:
            return y_sub



    def fit_predict(self, model, model_params={}, prep_params={}, fit_params={}, use_cols=None,
                    mean_func=None, return_oof=True):

        self.fit(model, model_params, prep_params, fit_params, use_cols)
        self.predict(mean_func, return_oof)



    def save(self, name=None):

        save_cv_results(self, name)
        return self



    def get_name(self):
        if hasattr(self, 'name'):
            return self.name
        elif hasattr(self, 'model_name'):
            return self.model_name
        else:
            return None


    def set_name(self, name):
        self.name = str(name)
        return self



    def fit_save(self, model, model_params={}, prep_params={}, fit_params={}, use_cols=None,
                 note=''):

        self.fit(model, model_params, prep_params, fit_params, use_cols)
        self.save(note)
        return self



    def load(self, idx):

        self = load_cv_results(self, idx)
        return self



    def load_multiple(self, indices=None, return_ind=False, debug_mode=False):
        cvs = []
        ind = []
        if indices is None:
            last_idx = last_file_idx()
            indices = range(last_idx+1)

        for idx in indices:
            try:
                cv = self.copy()
                cv.load(idx)
                cvs.append(cv)
                ind.append(idx)
            except Exception as e:
                if debug_mode:
                    print(e)

        if return_ind:
            return cvs, ind
        else:
            return cvs



    def copy(self):
        cv = copy.copy(self)
        cv.update_metric(self.metric_name)

        if hasattr(self, 'results'):
            cv.model_task = str(self.model_task)
            cv.model_name = str(self.model_name)
            if hasattr(self, 'name'):
                cv.set_name(self.name)

            cv.model_params = dict(self.model_params)
            cv.prep_params = dict(self.prep_params)
            cv.fit_params = dict(self.fit_params)
            cv.use_cols = list(self.use_cols)

            cv.results = list(self.results)

        return cv



    def pair_ztest(self, idx, show_table=False):

        cv = self.copy().load(idx)
        a, b = self.get_results(), cv.get_results()

        _, pval = ztest(a, b)

        if show_table:
            display_scores_table([self, cv])
        return pval



    def pair_ttest(self, idx, show_table=False):

        cv = self.copy().load(idx)
        a, b = self.get_results(), cv.get_results()

        if self.model_name == cv.model_name:
            # equal models, different params
            _, pval = ttest_rel(a, b)
        else:
            # different models
            _, pval = ttest_ind(a, b, equal_var=False)

        if show_table:
            display_scores_table([self, cv])
        return pval



    def lb_hist(self, indices=None, k_points=4, point='date', beta=2.0):
        # points = 'idx'/'date'/'time'
        cvs, ind = self.load_multiple(indices, return_ind=True)
        plot_scores_history(cvs, ind, point=point, k_points=k_points, beta=beta)


    def lb_rank(self, indices=None, k_top=10, fold=False):
        cvs, ind = self.load_multiple(indices, return_ind=True)
        display_scores_table(cvs, ind, sort=True, k_top=k_top, fold=fold)


    def lb_corr(self, indices=None, k_top=10, method='kendall', **sns_params):
        cvs, ind = self.load_multiple(indices, return_ind=True)
        display_corr(cvs, ind, k_top=k_top, method=method, **sns_params)


    def lb_tsne(self, indices=None, **tsne_params):
        cvs, ind = self.load_multiple(indices, return_ind=True)
        display_tsne(cvs, ind, **tsne_params)



    def stack(self, indices=None, original_cols=False):
        cvs, ind = self.load_multiple(indices, return_ind=True)
        y_oof, y_sub = stacking(cvs, ind)

        if original_cols and hasattr(self, 'X_train'):
            self.X_train = pd.concat([self.X_train, y_oof], axis=1)
            self.X_test = pd.concat([self.X_test, y_sub], axis=1)
        else:
            self.X_train = y_oof
            self.X_test = y_sub

        if not hasattr(self, 'folds'):
            self.split(self.X_train, self.X_test)

        return self


    def output(self):
        if self.plot:
            plot_fold(self)
        if self.verbose:
            print_fold(self)


'''
Cross-Validation Schemes
'''
from sklearn.model_selection import *


class KFoldCV(Validator):
    # k_folds

    # stratify
    # random_state

    def split(self, X_train, X_test, y_train=None, **data_params):

        self._split(X_train, X_test, y_train)

        cv_params = {
            'n_splits': 5,
            'shuffle': True,
            'stratify': (self.task == 'bin'),
            'random_state': 0
        }

        cv_params.update(self.cv_params)
        stratify = cv_params['stratify']
        cv_params.pop('stratify')

        splitter = StratifiedKFold(**cv_params) if stratify else KFold(**cv_params)
        self.folds = list(splitter.split(X_train, y_train))
        self.n_folds = splitter.get_n_splits()

        return self



class NKFoldCV(Validator):
    # k_folds
    # n_iters

    # stratify
    # random_state

    def split(self, X_train, X_test, y_train=None, **data_params):

        self._split(X_train, X_test, y_train)

        cv_params = {
            'n_splits': 5,
            'n_repeats': 3,
            'stratify': (self.task == 'bin'),
            'random_state': 0,
        }

        cv_params.update(self.cv_params)
        stratify = cv_params['stratify']
        cv_params.pop('stratify')

        splitter = RepeatedStratifiedKFold(**cv_params) if stratify else RepeatedKFold(**cv_params)
        self.folds = list(splitter.split(X_train, y_train))
        self.n_folds = splitter.get_n_splits()

        return self



class Holdout(Validator):
    # test_size
    # train_size

    # stratify
    # random_state

    def split(self, X_train, X_test, y_train=None, **data_params):

        self._split(X_train, X_test, y_train)

        cv_params = {
            'n_splits': 1,
            'test_size': 0.2,
            'train_size': None,
            'shuffle': True,
            'stratify': True,
            'random_state': 0,
        }

        cv_params.update(self.cv_params)
        stratify = cv_params['stratify']
        cv_params.pop('stratify')

        splitter = StratifiedKFold(**cv_params) if stratify else KFold(**cv_params)
        self.folds = list(splitter.split(X_train, y_train))
        self.n_folds = splitter.get_n_splits()

        return self



class LOO(Validator):

    def split(self, X_train, X_test, y_train=None, **data_params):

        self._split(X_train, X_test, y_train)

        splitter = LeaveOneOut()
        self.folds = list(splitter.split(X_train, y_train))
        self.n_folds = splitter.get_n_splits()

        return self



class GroupLOO(Validator):

    def split(self, X_train, X_test, y_train=None, **data_params):

        self._split(X_train, X_test, y_train)
        self.groups = data_params['groups']

        splitter = LeaveOneGroupOut()
        self.folds = list(splitter.split(X_train, y_train, groups))
        self.n_folds = splitter.get_n_splits()

        return self




class GroupKFoldCV(Validator):
    # k_folds

    # stratify
    # random_state

    def split(self, X_train, X_test, y_train=None, **data_params):

        self._split(X_train, X_test, y_train)
        self.groups = data_params['groups']

        cv_params = {
            'n_splits': 5,
        }

        splitter = GroupKFold(**cv_params)
        self.folds = list(splitter.split(X_train, y_train, groups))
        self.n_folds = splitter.get_n_splits()

        return self
