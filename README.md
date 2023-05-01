<p align="center">
    <img width=30% src="https://w7.pngwing.com/pngs/884/47/png-transparent-fight-club-thumbnail.png">
</p>

<!-- buttons -->
<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
            alt="python"></a> &nbsp;
    <a href="https://pypi.org/project/PyPortfolioOpt/">
        <img src="https://img.shields.io/badge/pypi-v1.0.0-brightgreen.svg"
            alt="pypi"></a> &nbsp;
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg"
            alt="MIT license"></a> &nbsp;
</p>

<!-- content -->
Developed by students of the [Simulator ML (Karpov.Courses)](https://karpov.courses/simulator-ml)

**Robusta ML Framework** is an extension of the Scikit-learn library that provides additional features and capabilities for data processing and building machine learning models.

Robusta ML Framework library features include:

-    Support for a large number of machine learning algorithms and models, including classical algorithms.
-   Implementation of data preprocessing methods such as feature scaling, outlier processing, categorical feature coding.
- Tools for choosing the best model, including cross-validation, hyperparameter fitting, and model evaluation.
- Ability to save and load results for later use.
    
## Table of contents


-   [Table of contents](#table-of-contents)
-   [Getting started](#getting-started)
    -   [For development](#for-development)
-   [A quick example](#a-quick-example)
-   [Features](#features)
    -   [Cross-validation](#cross-validation)
    -   [Importances](#importances)
    -   [Linear Models](#linear-models)
    -   [Optimizers](#optimizers)
-   [Project principles and design decisions](#project-principles-and-design-decisions)
-   [Testing](#testing)
-   [Getting in touch](#getting-in-touch)

## Getting started

This project is available on PyPI, meaning that you can just:

```bash
pip install robusta
```

Otherwise, clone/download the project and in the project directory run:

```bash
python setup.py install
```

### For development

If you would like to make major changes to integrate this with your proprietary system, it probably makes sense to clone this repository and to just use the source code.

```bash
git clone https://github.com/uberkinder/robusta
```

Alternatively, you could try:

```bash
pip install -e git+https://github.com/uberkinder/robusta.git
```

## A quick example

Here is an example on how to use extended cross-validation.

```python
# Basic
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter('ignore')

# ML Toolkit
from robusta.crossval import *
from robusta.pipeline import *
from robusta.preprocessing import *
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

# Data
from catboost.datasets import adult

# Model
from lightgbm import LGBMClassifier



TARGET = 'income'

train, test = adult()

# Target
labels_train = train['income']
labels_test = test['income']

train.drop(columns='income', inplace=True)
test.drop(columns='income', inplace=True)

# Target Binarization
y_train = labels_train.astype('category').cat.codes
y_test  = labels_test.astype('category').cat.codes

del labels_train, labels_test

prep_pipe = FeatureUnion([
    ("category", make_pipeline(
        TypeSelector("object"),
        Categorizer(),
    )),
    ("numeric", make_pipeline(
        TypeSelector(np.number),
    )),
])

X_train = prep_pipe.fit_transform(train)
X_test = prep_pipe.transform(test)

cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
model = LGBMClassifier()
scoring = 'roc_auc'

_, y_pred = crossval_predict(model, cv, X_train, y_train, X_new=X_test,
                             scoring=scoring, method='predict_proba',
                             verbose=2, n_jobs=None)

roc_auc_score(y_test, y_pred)
```

```txt
[20:23:34]  LGBMClassifier

[20:23:35]  FOLD  0:   0.9265

[20:23:35]  AVERAGE:   0.9265 ± 0.0000

0.9266619076504115
```

**Adversarial Validation**
```python
cv = make_adversarial_validation(model, X_train, X_test, test_size=0.2)
model = LGBMClassifier()
scoring = 'roc_auc'

_, y_pred = crossval_predict(model, cv, X_train, y_train, X_new=X_test,
                             scoring=scoring, method='predict_proba',
                             verbose=2, n_jobs=None)

roc_auc_score(y_test, y_pred)
```

```txt
[20:24:05]  LGBMClassifier

[20:24:05]  FOLD  0:   0.9328

[20:24:06]  AVERAGE:   0.9328 ± 0.0000

0.9260441555579393
```

## Features

In this section, we detail some of robusta available functionality. More examples are offered in the Jupyter notebooks [here](https://github.com/nikneural/robust/tree/master/cookbook). Another good resource is the [tests](https://github.com/nikneural/robust/tree/master/tests).

### Cross-validation

-   RepeatedGroupKFold
-   RepeatedKFold
-   StratifiedGroupKFold
-   RepeatedStratifiedGroupKFold
-   AdversarialValidation

### Importances

-   PermutationImportance
-   GroupPermutationImportance
-   ShuffleTargetImportance

### Linear Models

-   BlendRegressor
-   BlendClassifier
-   CaruanaRegressor
-   NNGRegressor

### Optimizers

-   GridSearchCV
-   OptunaCV
-   RandomSearchCV


## Project principles and design decisions

-   It should be easy to swap out individual components of the optimization process
    with the user's proprietary improvements.
-   Usability is everything: it is better to be self-explanatory than consistent.
-   There is no point in portfolio optimization unless it can be practically
    applied to real asset prices.
-   Everything that has been implemented should be tested.
-   Inline documentation is good: dedicated (separate) documentation is better.
    The two are not mutually exclusive.
-   Formatting should never get in the way of coding: because of this,
    I have deferred **all** formatting decisions to [Black](https://github.com/ambv/black).
    
## Testing

Tests are written in pytest, and I have tried to ensure close to 100% coverage. Run the tests by navigating to the package directory and simply running `pytest` on the command line.



## Getting in touch

If you are having a problem with Robusta, please raise a GitHub issue.
