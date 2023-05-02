<!-- content -->
Developed by students of the [Simulator ML (Karpov.Courses)](https://karpov.courses/simulator-ml)

**Robusta ML Framework** is an extension of the Scikit-learn library that provides additional features and capabilities for data processing and building machine learning models.

Robusta ML Framework library features include:

- Support for a large number of machine learning algorithms and models, including classical algorithms.
- Implementation of data preprocessing methods such as feature scaling, outlier processing, categorical feature coding.
- Tools for choosing the best model, including cross-validation, hyperparameter fitting, and model evaluation.
- Ability to save and load results for later use.
    
## Table of contents


-   [Table of contents](#table-of-contents)
-   [Getting started](#getting-started)
    -   [For development](#for-development)
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
