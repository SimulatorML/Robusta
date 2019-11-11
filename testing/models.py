import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

__all__ = ['MODELS']




COLUMNS = ['name', 'model', 'tags']
MODELS = []

def _import_models(lib, tags=set()):
    for name in lib.__all__:
        estimator = getattr(lib, name)
        if hasattr(estimator, 'fit'):
            yield [name, estimator, tags]


'''Scikit-Learn Linear Model
https://scikit-learn.org/stable/modules/linear_model
'''
from sklearn import linear_model

models = _import_models(linear_model, {'linear'})
MODELS.extend(models)


'''Blend, NNG & etc
'''
from robusta import linear_model

models = _import_models(linear_model, {'linear'})
MODELS.extend(models)


'''MARS (Multivariate Adaptive Regression Splines)
https://contrib.scikit-learn.org/py-earth/
'''
from pyearth import Earth

MODELS.append(['MARS', Earth, {'linear'}])


'''RVM (Relevance Vector Machine)
https://github.com/JamesRitchie/scikit-rvm
'''
from skrvm import RVR, RVC

MODELS.append(['RVM', RVR, {'svm', 'proba'}])
MODELS.append(['RVM', RVC, {'svm', 'proba'}])


'''Scikit-Learn SVM
https://scikit-learn.org/stable/modules/naive_bayes.html
'''
from sklearn.svm import *

MODELS.append(['SVM', SVR, {'svm'}])
MODELS.append(['SVM', SVC, {'svm'}])
MODELS.append(['NuSVM', SVR, {'svm'}])
MODELS.append(['NuSVM', SVC, {'svm'}])
MODELS.append(['LinearSVM', SVR, {'linear', 'svm'}])
MODELS.append(['LinearSVM', SVC, {'linear', 'svm'}])
MODELS.append(['OneClassSVM', OneClassSVM, {'svm'}])


'''Scikit-Learn Disctiminant Analysis
https://scikit-learn.org/stable/modules/discriminant_analysis
'''
from sklearn.discriminant_analysis import *

MODELS.append(['LDA', LinearDiscriminantAnalysis, {'linear'}])
MODELS.append(['QDA', QuadraticDiscriminantAnalysis, {'linear'}])


'''Scikit-Learn Nearest Neighbors
https://scikit-learn.org/stable/modules/linear_model
'''
from sklearn import neighbors

models = _import_models(neighbors, {'neighbors', 'dense'})
MODELS.extend(models)


'''Scikit-Learn Gaussian Processes
https://scikit-learn.org/stable/modules/gaussian_process
'''
from sklearn import gaussian_process

models = _import_models(gaussian_process, {'proba', 'dense'})
MODELS.extend(models)


'''Scikit-Learn TreeBoost
https://scikit-learn.org/stable/modules/ensemble
'''
from sklearn import ensemble

models = _import_models(ensemble, {'tree', 'ensemble'})
MODELS.extend(models)


'''Scikit-Learn Decision Tree
https://scikit-learn.org/stable/modules/tree
'''
from sklearn import tree

models = _import_models(tree, {'tree'})
MODELS.extend(models)


'''CatBoost, LightGBM, XGBoost
https://xgboost.readthedocs.io
https://lightgbm.readthedocs.io/
https://catboost.ai/docs/
'''
from xgboost import *
from lightgbm import *
from catboost import *

TAGS_ = {'tree', 'ensemble'}

MODELS.append(['XGB', XGBRegressor, TAGS_])
MODELS.append(['XGB', XGBClassifier, TAGS_])
MODELS.append(['XGB', XGBRanker, TAGS_])

MODELS.append(['LGB', LGBMClassifier, TAGS_])
MODELS.append(['LGB', LGBMRegressor, TAGS_])
MODELS.append(['LGB', LGBMRanker, TAGS_])

# TODO: Wrapper for CatBoost
# 1. no verbose by default
# 2. no cached files
# 3. _estimator_type
# 4. __repr__ (via BaseEstimator)
#MODELS.append(['CatBoost', CatBoostClassifier, TAGS_])
#MODELS.append(['CatBoost', CatBoostRegressor, TAGS_])


'''BART (Bayesian Additive Regressions Trees)
https://github.com/JakeColtman/bartpy
'''
from bartpy.sklearnmodel import SklearnModel as BART

TAGS_ = {'tree', 'ensemble'}

MODELS.append(['BART', BART, TAGS_])


#'''RGF (Regularized Greedy Forest)
#https://github.com/RGF-team/rgf/tree/master/python-package
#'''
#from rgf.sklearn import *

#TAGS_ = {'tree', 'ensemble'}

#MODELS.append(['RGF', RGFClassifier, TAGS_])
#MODELS.append(['RGF', RGFRegressor, TAGS_])
#MODELS.append(['FastRGF', FastRGFClassifier, TAGS_])
#MODELS.append(['FastRGF', FastRGFRegressor, TAGS_])


'''Hierarchical Density Based Clustering
https://hdbscan.readthedocs.io/
'''
from hdbscan import HDBSCAN

MODELS.append(['HDBSCAN', HDBSCAN, {'dense'}])


'''Scikit-Learn Cluster
https://scikit-learn.org/stable/modules/mixture
'''
from sklearn import mixture

models = _import_models(mixture, {'dense'})
MODELS.extend(models)


'''Scikit-Learn Cluster
https://scikit-learn.org/stable/modules/cluster
'''
from sklearn import cluster

models = _import_models(cluster, {'dense'})
MODELS.extend(models)


'''Imbalanced Learn Resampling

'''
import imblearn

def _add_tags(name):
    if 'Neighbour' in name:
        yield 'neighbors'
    if 'KNN' in name:
        yield 'neighbors'
    if 'SVM' in name:
        yield 'svm'

def _import_samplers(lib, tags=set()):
    for name in lib.__all__:
        sampler = getattr(lib, name)
        if hasattr(sampler, 'fit'):
            #sampler = make_sampler(sampler)
            tags_ = set(_add_tags(name))
            yield [name, sampler, tags_ | tags]

for lib in [imblearn.over_sampling,
            imblearn.under_sampling,
            imblearn.combine]:
    samplers = _import_samplers(lib)
    MODELS.extend(samplers)


'''
POST-PROCESSING
'''

# create DataFrame
MODELS = pd.DataFrame(MODELS, columns=COLUMNS)

# extract estimator's type
ETYPES = [
    'classifier',
    'regressor',
    'clusterer',
    'outlier_detector',
    'ranker',
    'sampler',
    'DensityEstimator',
]

def _extract_type(c):
    if hasattr(c, '_estimator_type'):
        return getattr(c, '_estimator_type')
    else:
        name = c.__name__
        for etype in ETYPES:
            if etype in name.lower():
                return etype
        return None

MODELS['type'] = MODELS['model'].map(_extract_type)


# drop type from name
DROPNAMES = ['Regressor', 'Classifier', 'Ranker']

def _drop_type(name):
    for s in DROPNAMES:
        name = name.replace(s, '')
    return name

MODELS['name'] = MODELS['name'].map(_drop_type)


# reorder columns
MODELS = MODELS[['name', 'type', 'model', 'tags']]
