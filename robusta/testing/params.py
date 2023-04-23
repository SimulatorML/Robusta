PARAM_SPACE = {
    'XGB': {
    # https://xgboost.readthedocs.io/en/latest/parameter.html
        'learning_rate': {0.3},
        # Once your learning rate is fixed, do not change it.

        'n_estimators': (100, 2000, 100),

        'max_depth': (3, 10, 1),
        'max_leaves': {15, 31, 63, 127, 255, 511, 1023, 2047, 4095},

        'subsample': (0.1, 0.9, 0.1),
        'colsample_bytree': (0.1, 0.9, 0.1),
        'colsample_bylevel': (0.1, 0.9, 0.1),

        #'gamma': (1e-6, 1e6, 'log'),
        #'alpha': (1e-6, 1e6, 'log'),
        #'lambda': (1e-6, 1e6, 'log'),
    },

    'LGB': {
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
        'learning_rate': {0.3},
        # Once your learning rate is fixed, do not change it.

        'n_estimators': (100, 2000, 100),

        'max_depth': (3, 10, 1),
        'num_leaves': {15, 31, 63, 127, 255, 511, 1023, 2047, 4095},

        'bagging_fraction': (0.1, 0.9, 0.1),
        'feature_fraction': (0.1, 0.9, 0.1),

        #'lambda_l1': (1e-6, 1e6, 'log'),
        #'lambda_l2': (1e-6, 1e6, 'log'),
    },

    'CatBoost': {
    # https://catboost.ai/docs/concepts/parameter-tuning.html
        #'learning_rate': None,
        # By default, the learning rate is defined automatically based on the dataset properties and the number of iterations.
        # The automatically defined value should be close to the optimal one.

        'n_estimators': (100, 3000, 100),

        'depth': (3, 10, 1),
        'l2_leaf_reg': (1e-6, 1e6, 'log'),

        'bagging_temperature': (1e-6, 1e6, 'log'),
        'random_strength': (1, 1e1, 'log'),
    },

    'MARS': {
    # https://github.com/scikit-learn-contrib/py-earth/blob/master/pyearth/earth.py
        #'max_terms': (1, 5, 1),
        #'max_degree': {1, 2},
        'penalty': (1e-6, 1e6, 'log'),
        #'endspan': (1, 30, 1),

        'enable_pruning': {True, False},
        'allow_missing': True,
        'allow_linear': {True, False},
    },

    'RGF': {
    # https://github.com/RGF-team/rgf/tree/master/python-package
        'algorithm': {"RGF", "RGF_Opt", "RGF_Sib"},
        'loss': {"LS", "Log", "Expo", "Abs"},

        'test_interval': {10, 50, 100, 200, 500, 1000},
        # For efficiency, it must be either multiple or divisor of 100.

        'max_leaf': {1000, 2000, 5000, 10000},
        # Appropriate values are data-dependent and vary from 1000 to 10000.

        'reg_depth': (1, 10, 1),

        'l2': {1, 0.1, 0.01},
        # Either 1, 0.1, or 0.01 often produces good results though with
        # exponential loss (loss='Expo') and logistic loss (loss='Log')
        # some data requires smaller values such as 1e-10 or 1e-20

        'sl2': {1, 0.1, 0.01},
        # By default equal to l2. On some data, l2/100 works well
    },

    'KNeighbors': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        'n_neighbors': (1, 30, 1),
        'weights': {'uniform', 'distance'},
        'algorithm': {'ball_tree', 'kd_tree'},
        'leaf_size': (10, 100, 10),
        'p': {1, 2},
    },

    'RadiusNeighbors': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html
        'radius': (1e-3, 1e2, 'log'),
        'weights': {'uniform', 'distance'},
        'algorithm': {'ball_tree', 'kd_tree'},
        'leaf_size': (10, 100, 10),
        'p': {1, 2},
    },

    'NearestCentroid': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html
        'metric': {'euclidean', 'manhattan'},
    },

    'LinearSVM': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
        'dual': {True, False},
        'loss': {'hinge', 'squared_hinge'},

        'penalty': {'l1', 'l2'},

        'C': (1e-6, 1e6, 'log'),

        #'max_iter': (10000, 50000, 5000),

        'probability': {True},
    },

    'SVM': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        'C': (1e-3, 1e2, 'log'),
        'gamma': (1e-3, 1e2, 'log'),

        #'kernel': 'rbf',
        #'degree': {2, 3},

        #'shrinking': {True, False},

        #'probability': {True, False},
    },

    'NuSVM': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html
        'nu': (1e-6, 1),
        'gamma': (1e-3, 1e2, 'log'),

        #'kernel': {'linear', 'rbf', 'sigmoid'},
        #'kernel': 'rbf',
        #'kernel': {'linear', 'poly', 'rbf', 'sigmoid'},
        #'degree': {2, 3},

        #'shrinking': {True, False},

        #'probability': {True, False},
    },

    'RVM': {
    # https://github.com/JamesRitchie/scikit-rvm
        'kernel': {'rbf'},
        #'kernel': {'linear', 'poly', 'rbf'},
        #'degree': {2, 3},
        #'degree': 2,

        'alpha': (1e-6, 1, 'log'),
        'beta': (1e-6, 1, 'log'),

        'coef0': (1e-6, 1, 'log'),
        'coef1': (1e-6, 1, 'log'),
    },

    'GradientBoosting': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        'n_estimators': (100, 2000, 100),
        'learning_rate': (1e-6, 1, 'log'),

        'subsample': (0.1, 1),
        'max_depth': (3, 16, 1),

        'loss': {'deviance', 'exponential'},
        'criterion': {'mse', 'friedman_mse', 'mae'},
        'max_features': {'auto', 'sqrt', 'log2', None},
    },

    'AdaBoost': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
        'n_estimators': (100, 2000, 100),
        'learning_rate': (1e-6, 1, 'log'),

        'algorithm': {'SAMME.R', 'SAMME'},
    },

    'RandomForest': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        'n_estimators': (100, 5000, 100),

        'min_samples_split': (1e-6, 1e-3, 'log'),
        'max_features': {'auto', 'sqrt', 'log2', None},

        'bootstrap': {True, False},
        'oob_score': {True},
    },

    'ExtraTrees': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
        'n_estimators': (100, 5000, 100),

        'min_samples_split': (1e-6, 1e-3, 'log'),
        'max_features': {'auto', 'sqrt', 'log2', None},

        'bootstrap': {True, False},
        'oob_score': {True},
    },

    'LinearRegression': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        'fit_intercept': {True, False},
    },

    'LogisticRegression': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        'penalty': {'l1', 'l2'},
        'C': (1e-6, 1e6, 'log'),

        'dual': {True, False},
        'fit_intercept': {True, False},

        'solver': {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
        'max_iter': {1000},
    },

    'SGD': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        'loss': {'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'},
        'fit_intercept': {True, False},

        'penalty': {'none', 'l2', 'l1', 'elasticnet'},
        'alpha': (1e-6, 1e6, 'log'),
        'l1_ratio': (0.0, 1.0),
        'epsilon': (1e-6, 1, 'log'),

        'learning_rate': {'constant', 'optimal', 'invscaling', 'adaptive'},
        'eta0': (1e-6, 1, 'log'),
        'power_t': (0, 2),
        'max_iter': (1000, 10000, 1000),
    },

    'PassiveAggressive': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html
        'C': (1e-6, 1e6, 'log'),
        'max_iter': (1000, 10000, 1000),

        'fit_intercept': {True, False},
        'loss': {'hinge', 'squared_hinge'},
    },

    'Ridge': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
        'alpha': (1e-6, 1e6, 'log'),

        'fit_intercept': {True, False},
        'normalize': {True, False},

        #'solver': {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'},
    },

    'Lasso': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        'alpha': (1e-6, 1e6, 'log'),

        'fit_intercept': {True, False},
        'normalize': {True, False},
        'positive': {True, False},

        'selection': {'cyclic', 'random'},
    },

    'LassoLars': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html
        'alpha': (1e-6, 1e6, 'log'),
        'eps': (1e-6, 1e-2, 'log'),

        'fit_intercept': {True, False},
        'normalize': {True, False},
        'positive': {True, False},
    },

    'Lars': {
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html
        'eps': (1e-6, 1e-2, 'log'),

        #'n_nonzero_coefs': lambda X: (1, np.shape(X)[1], 1),

        'fit_intercept': {True, False},
        'normalize': {True, False},
        'positive': {True, False},
    },

    'BART': {
    # https://github.com/JakeColtman/bartpy/blob/master/bartpy/sklearnmodel.py
        'n_trees': (100, 2000, 100),
        'n_chains': (1, 32, 1),

        'sigma_a': (1e-6, 1, 'log'),
        'sigma_b': (1e-6, 1, 'log'),

        'thin': (0, 1),
        'p_grow': (0, 1),
        'p_prune': (0, 1),

        'alpha': (0, 1),
        'beta': (0, 1e5, 'log'),

        'store_in_sample_predictions': {False},
    },

}
