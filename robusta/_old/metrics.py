from sklearn.metrics import *
from robusta import utils
import numpy as np

def needs_proba(eval_metric): return utils.get_params(eval_metric)[1] == 'y_prob'

def auc_score(y_true, y_prob): return roc_auc_score(y_true, y_prob)
def logloss_score(y_true, y_prob): return -log_loss(y_true, y_prob)
def rmsle_score(y_true, y_pred): return -np.sqrt(mean_squared_log_error(y_true, y_pred))
def rmse_score(y_true, y_pred): return -np.sqrt(mean_squared_error(y_true, y_pred))
def mae_score(y_true, y_pred): return -mean_absolute_error(y_true, y_pred)

def map_score(k):
    return 'MAP@%d' % k

def mnap_score(k):
    return 'MNAP@%d' % k



def get_metric(eval_metric, return_task=False, return_color=False):
    if '@' in eval_metric:
        splitting = eval_metric.split('@')
        metric_name = splitting[0]
        params_vals = splitting[1:]
        params_vals = np.vectorize(float)(params_vals)
        params_vals = utils.int_list(params_vals)
        
        metric_keys = list(metric_task)
        metric_names = np.vectorize(lambda x: x.split('@')[0])(metric_keys)
        metric_key = metric_keys[np.where(metric_names == metric_name)[0][0]]
        
        params_args = metric_key.split('@')[1:]
        params = dict(zip(params_args, params_vals))
        
        task = metric_task[metric_key]
        color = metric_color[metric_key]
        metric = eval_metrics[task][metric_key](**params)
        
    else:
        task = metric_task[eval_metric]
        color = metric_color[eval_metric]
        metric = eval_metrics[task][eval_metric]
        
    result = (metric,)
    if return_task: 
        result += (task,)
    if return_color: 
        result += (color,)
    if len(result) == 1:
        result = result[0]
    return result

            

eval_metrics = {
    'bin': {
        'Accuracy': accuracy_score, 
        'BalancedAccuracy': balanced_accuracy_score,
        'Recall': recall_score,
        'Precision': precision_score, 
        'F1': f1_score,
        'AUC': auc_score, #proba
        'LogLoss': logloss_score, #proba
    },
    'reg': {
        'R2': r2_score,
        'MAE': mae_score,
        'RMSE': rmse_score,
        'RMSLE': rmsle_score,
    },
    'rank': {
        'MAP@k': lambda k: map_score(k),
        'MNAP@k': lambda k: mnap_score(k),
    }
}


metric_color = {
    'Accuracy': 'y', # orange
    'BalancedAccuracy': 'y', # orange
    'Recall': 'r',
    'Precision': 'g',
    'F1': 'b',
    'AUC': '#f84b3c', # red
    'LogLoss': 'c', # light blue
    'R2': '#f84b3c', # red
    'MAE': '#ff8642', # orange
    'RMSE': 'y', 
    'RMSLE': '#fbb339', # yellow
    'MAP@k': 'green',
    'MNAP@k': 'violet'
}


metric_task = {}
for task, metrics in eval_metrics.items():
    for metric_name in metrics:
        metric_task[metric_name] = task