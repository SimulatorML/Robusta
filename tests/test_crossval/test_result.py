import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from robusta.crossval import argsort_idx, check_cvs, split_cv_groups, rating_table


def test_argsort_idx():
    # Test the function with a simple example
    idx_list = [2.0, 1.3, 3.5, 1.7, 2.4]
    expected_result = [1, 3, 0, 4, 2]
    assert np.all(argsort_idx(idx_list) == expected_result)

    # Test the function with a list of identical elements
    idx_list = [1.2] * 5
    expected_result = [0, 1, 2, 3, 4]
    assert np.all(argsort_idx(idx_list) == expected_result)

def test_check_cvs():
    # create sample data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    groups = np.array([0, 0, 1, 1])

    # create sample cv objects
    cv1 = KFold(n_splits=2)
    cv2 = KFold(n_splits=2, shuffle=True)

    # create sample results list
    results = [{'cv': cv1}, {'cv': cv2}]

    # modify cv2 to have the same splits as cv1
    cv2.split = cv1.split
    cv2.get_n_splits = cv1.get_n_splits

    # test that function returns the cv object when all cv objects have the same splits
    assert check_cvs(results, X, y, groups) == cv1

@pytest.fixture
def results():
    return {
        'result1': {'cv': 5},
        'result2': {'cv': 10},
        'result3': {'cv': 3}
    }

@pytest.fixture
def y_train():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def test_split_cv_groups(results, y_train):
    cv_groups = split_cv_groups(results, y_train)
    print(cv_groups[0])
    assert len(cv_groups) == 3
    assert len(cv_groups[0]) == 1
    assert len(cv_groups[1]) == 1

def test_rating_table():
    results = {
        1: {'model_name': 'Model A', 'public_score': 0.8, 'private_score': 0.7, 'val_score': [0.9, 0.8, 0.7]},
        2: {'model_name': 'Model B', 'public_score': 0.7, 'private_score': 0.6, 'val_score': [0.8, 0.7, 0.6]},
        3: {'model_name': 'Model C', 'public_score': 0.6, 'private_score': 0.5, 'val_score': [0.7, 0.6, 0.5]},
    }
    expected_output = pd.DataFrame({
        'MODEL_NAME': ['Model A', 'Model B', 'Model C'],
        'PRIVATE': [0.7, 0.6, 0.5],
        'PUBLIC': [0.8, 0.7, 0.6],
        'LOCAL': [0.8, 0.7, 0.6],
        'STD': [0.1, 0.1, 0.1],
        'MIN': [0.7, 0.6, 0.5],
        'MAX': [0.9, 0.8, 0.7]
    })
    expected_output = expected_output.set_index(pd.Index([1, 2, 3]))

    output = rating_table(results, n_digits=1)
    pd.testing.assert_frame_equal(output, expected_output)