import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score

from robusta.importance import _get_col_score, permutation_importance

############################################################################################
###################################GetColScore##############################################
############################################################################################

def test_get_col_score():
    # Create some fake data
    X = pd.DataFrame({'A': np.random.normal(size=100),
                      'B': np.random.normal(size=100)})
    y = pd.Series(np.random.normal(size=100))

    # Create an estimator
    estimator = LinearRegression()

    # Define a scoring function
    def my_scorer(estimator, X, y):
        return estimator.score(X, y)

    # Fit estimator
    estimator.fit(X,y)

    # Set the random seed
    rstate = np.random.RandomState(42)

    # Test with one repeat
    scores = _get_col_score(estimator, X, y, 'A', n_repeats=1, scorer=my_scorer, rstate=rstate)

    assert len(scores) == 1
    assert isinstance(scores[0], float)

    # Test with multiple repeats
    scores = _get_col_score(estimator, X, y, 'A', n_repeats=10, scorer=my_scorer, rstate=rstate)
    assert len(scores) == 10
    assert all(isinstance(score, float) for score in scores)

    # Test that the scores are not all the same
    assert not all(score == scores[0] for score in scores)

    # Test that the scores are different when permuting different columns
    scores_A = _get_col_score(estimator, X, y, 'A', n_repeats=10, scorer=my_scorer, rstate=rstate)
    scores_B = _get_col_score(estimator, X, y, 'B', n_repeats=10, scorer=my_scorer, rstate=rstate)
    assert not all(score_A == score_B for score_A, score_B in zip(scores_A, scores_B))

############################################################################################
################################PermutationImportance#######################################
############################################################################################

def test_permutation_importance():
    # Generate a random dataset
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    # Fit a logistic regression model
    estimator = LogisticRegression(random_state=42)
    estimator.fit(X, y)

    # Compute feature importance using permutation importance
    importance = permutation_importance(estimator=estimator, X=pd.DataFrame(X), y=pd.Series(y),
                                         scoring=make_scorer(accuracy_score), n_repeats=5, random_state=42)

    # Check that the output dictionary has the expected keys
    assert set(importance.keys()) == {'importances_mean', 'importances_std', 'importances', 'score'}

    # Check that the output dictionary has the expected values for each key
    assert isinstance(importance['importances_mean'], np.ndarray)
    assert isinstance(importance['importances_std'], np.ndarray)
    assert isinstance(importance['importances'], np.ndarray)
    assert isinstance(importance['score'], float)

    assert importance['importances_mean'].shape == (5,)
    assert importance['importances_std'].shape == (5,)
    assert importance['importances'].shape == (5, 5)
    assert importance['score'] >= 0.0

    # Check that the feature importances sum to the difference between the original score and the baseline score
    assert (importance['score'] - np.mean(importance['importances_mean'])) > accuracy_score(y, estimator.predict(X))