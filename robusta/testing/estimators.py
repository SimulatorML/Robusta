import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


__all__ = ["ESTIMATORS"]


COLUMNS = ["name", "class", "tags"]
ESTIMATORS = []


def _import_models(lib, tags=set()):
    for name in lib.__all__:
        estimator = getattr(lib, name)
        if hasattr(estimator, "fit"):
            yield [name, estimator, tags]


"""Scikit-Learn Linear Model
https://scikit-learn.org/stable/modules/linear_model
"""
from sklearn import linear_model

estimators = _import_models(linear_model, {"linear"})
ESTIMATORS.extend(estimators)


"""Blend, NNG & etc
"""
from robusta import linear_model

estimators = _import_models(linear_model, {"linear"})
ESTIMATORS.extend(estimators)


"""MARS (Multivariate Adaptive Regression Splines)
https://contrib.scikit-learn.org/py-earth/
"""
# from pyearth import Earth

# class MARS(Earth):
#    pass

# ESTIMATORS.append(['MARS', MARS, {'linear'}])


"""RVM (Relevance Vector Machine)
https://github.com/JamesRitchie/scikit-rvm
"""
# from skrvm import RVR, RVC

# ESTIMATORS.append(['RVM', RVR, {'svm', 'proba'}])
# ESTIMATORS.append(['RVM', RVC, {'svm', 'proba'}])


"""Scikit-Learn SVM
https://scikit-learn.org/stable/modules/naive_bayes.html
"""
from sklearn.svm import *

ESTIMATORS.append(["SVM", SVR, {"svm"}])
ESTIMATORS.append(["SVM", SVC, {"svm"}])
ESTIMATORS.append(["NuSVM", SVR, {"svm"}])
ESTIMATORS.append(["NuSVM", SVC, {"svm"}])
ESTIMATORS.append(["LinearSVM", SVR, {"linear", "svm"}])
ESTIMATORS.append(["LinearSVM", SVC, {"linear", "svm"}])
ESTIMATORS.append(["OneClassSVM", OneClassSVM, {"svm"}])


"""Scikit-Learn Disctiminant Analysis
https://scikit-learn.org/stable/modules/discriminant_analysis
"""
from sklearn.discriminant_analysis import *


class LDA(LinearDiscriminantAnalysis):
    pass


class QDA(QuadraticDiscriminantAnalysis):
    pass


ESTIMATORS.append(["LDA", LDA, {"linear"}])
ESTIMATORS.append(["QDA", QDA, {"linear"}])


"""Scikit-Learn Nearest Neighbors
https://scikit-learn.org/stable/modules/linear_model
"""
from sklearn import neighbors

estimators = _import_models(neighbors, {"neighbors", "dense"})
ESTIMATORS.extend(estimators)


"""Scikit-Learn Gaussian Processes
https://scikit-learn.org/stable/modules/gaussian_process
"""
from sklearn import gaussian_process

estimators = _import_models(gaussian_process, {"proba", "dense"})
ESTIMATORS.extend(estimators)


"""Scikit-Learn TreeBoost
https://scikit-learn.org/stable/modules/ensemble
"""
from sklearn import ensemble

estimators = _import_models(ensemble, {"tree", "ensemble"})
ESTIMATORS.extend(estimators)


"""Scikit-Learn Decision Tree
https://scikit-learn.org/stable/modules/tree
"""
from sklearn import tree

estimators = _import_models(tree, {"tree"})
ESTIMATORS.extend(estimators)


"""CatBoost, LightGBM, XGBoost
https://xgboost.readthedocs.io
https://lightgbm.readthedocs.io/
https://catboost.ai/docs/
"""
from xgboost import *
from lightgbm import *
from catboost import *

TAGS_ = {"tree", "ensemble"}

ESTIMATORS.append(["XGB", XGBRegressor, TAGS_])
ESTIMATORS.append(["XGB", XGBClassifier, TAGS_])
ESTIMATORS.append(["XGB", XGBRanker, TAGS_])

ESTIMATORS.append(["LGB", LGBMClassifier, TAGS_])
ESTIMATORS.append(["LGB", LGBMRegressor, TAGS_])
ESTIMATORS.append(["LGB", LGBMRanker, TAGS_])

# TODO: Wrapper for CatBoost
# 1. no verbose by default
# 2. no cached files


class CatBoostClassifier(ClassifierMixin, BaseEstimator, CatBoostClassifier):
    pass


class CatBoostRegressor(RegressorMixin, BaseEstimator, CatBoostRegressor):
    pass


ESTIMATORS.append(["CatBoost", CatBoostClassifier, TAGS_])
ESTIMATORS.append(["CatBoost", CatBoostRegressor, TAGS_])


"""BART (Bayesian Additive Regressions Trees)
https://github.com/JakeColtman/bartpy
"""
# from bartpy.sklearnmodel import SklearnModel

# class BART(SklearnModel):
#    pass

# TAGS_ = {'tree', 'ensemble'}

# ESTIMATORS.append(['BART', BART, TAGS_])


#'''RGF (Regularized Greedy Forest)
# https://github.com/RGF-team/rgf/tree/master/python-package
#'''
# from rgf.sklearn import *

# TAGS_ = {'tree', 'ensemble'}

# ESTIMATORS.append(['RGF', RGFClassifier, TAGS_])
# ESTIMATORS.append(['RGF', RGFRegressor, TAGS_])
# ESTIMATORS.append(['FastRGF', FastRGFClassifier, TAGS_])
# ESTIMATORS.append(['FastRGF', FastRGFRegressor, TAGS_])


"""Hierarchical Density Based Clustering
https://hdbscan.readthedocs.io/
"""
from hdbscan import HDBSCAN

ESTIMATORS.append(["HDBSCAN", HDBSCAN, {"dense"}])


"""Scikit-Learn Cluster
https://scikit-learn.org/stable/modules/mixture
"""
from sklearn import mixture

estimators = _import_models(mixture, {"dense"})
ESTIMATORS.extend(estimators)


"""Scikit-Learn Cluster
https://scikit-learn.org/stable/modules/cluster
"""
from sklearn import cluster

estimators = _import_models(cluster, {"dense"})
ESTIMATORS.extend(estimators)


"""Robusta Transfomrers
"""
from robusta.preprocessing import base, numeric, category, target

estimators = _import_models(base, {})
ESTIMATORS.extend(estimators)

estimators = _import_models(numeric, {"numeric"})
ESTIMATORS.extend(estimators)

estimators = _import_models(category, {"category"})
ESTIMATORS.extend(estimators)

estimators = _import_models(target, {"category", "target"})
ESTIMATORS.extend(estimators)


"""Imbalanced Learn Resampling
https://imbalanced-learn.org/stable/api.html
"""
import imblearn


def _add_tags(name):
    if "Neighbour" in name:
        yield "neighbors"
    if "KNN" in name:
        yield "neighbors"
    if "SVM" in name:
        yield "svm"


def _import_samplers(lib, tags=set()):
    for name in lib.__all__:
        sampler = getattr(lib, name)
        if hasattr(sampler, "fit"):
            # sampler = make_sampler(sampler)
            tags_ = set(_add_tags(name))
            yield [name, sampler, tags_ | tags]


for lib in [imblearn.over_sampling, imblearn.under_sampling, imblearn.combine]:
    samplers = _import_samplers(lib)
    ESTIMATORS.extend(samplers)


"""
POST-PROCESSING
"""

# create DataFrame
ESTIMATORS = pd.DataFrame(ESTIMATORS, columns=COLUMNS)

# extract estimator's type
ESTIMATOR_TYPES = [
    "classifier",
    "regressor",
    "clusterer",
    "outlier_detector",
    "ranker",
    "sampler",
    "DensityEstimator",
]


def _estimator_type(estimator):
    if hasattr(estimator, "_estimator_type"):
        return getattr(estimator, "_estimator_type")

    elif hasattr(estimator, "transform"):
        return "transformer"

    else:
        name = estimator.__name__
        for estimator_type in ESTIMATOR_TYPES:
            if estimator_type in name.lower():
                return estimator_type
        return None


ESTIMATORS["type"] = ESTIMATORS["class"].map(_estimator_type)


# drop type from name
DROPNAMES = ["Regressor", "Classifier", "Ranker"]


def _drop_type(name):
    for s in DROPNAMES:
        name = name.replace(s, "")
    return name


ESTIMATORS["name"] = ESTIMATORS["name"].map(_drop_type)

# full name
ESTIMATORS["class_name"] = ESTIMATORS["class"].map(lambda x: x.__name__)


# reorder columns
ESTIMATORS = ESTIMATORS[["class", "class_name", "name", "type", "tags"]]
