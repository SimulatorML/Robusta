import numpy as np
import pandas as pd
import pytest

from robusta.selector import _check_max_features, _check_threshold

############################################################################################
################################CheckMaxFeatures############################################
############################################################################################

def test_check_max_features():
    # Create a sample Pandas Series with feature importances
    importances = pd.Series([0.1, 0.2, 0.3, 0.4])

    # Test if max_features=None, all features are selected
    assert _check_max_features(importances, None) == 4

    # Test if max_features is an integer, correct number of features are selected
    assert _check_max_features(importances, 2) == 2
    assert _check_max_features(importances, 5) == 4  # can't select more than total features

    # Test if max_features is a float, correct number of features are selected
    assert _check_max_features(importances, 0.5) == 2  # half of 4 features is 2
    assert _check_max_features(importances, 0.75) == 3  # 3/4 of 4 features is 3

############################################################################################
################################CheckThreshold##############################################
############################################################################################

@pytest.fixture(scope="module")
def importances():
    return pd.Series([0.2, 0.3, 0.5, 0.1])

def test_check_threshold_none(importances):
    threshold = None
    expected_result = -np.inf
    assert _check_threshold(importances, threshold) == expected_result

def test_check_threshold_string_median(importances):
    threshold = "median"
    expected_result = 0.25
    assert _check_threshold(importances, threshold) == expected_result

def test_check_threshold_string_mean(importances):
    threshold = "mean"
    expected_result = 0.275
    assert _check_threshold(importances, threshold) == expected_result

def test_check_threshold_string_scale_median(importances):
    threshold = "2 * median"
    expected_result = 0.5
    assert _check_threshold(importances, threshold) == expected_result

def test_check_threshold_string_scale_mean(importances):
    threshold = "0.5 * mean"
    expected_result = 0.1375
    assert _check_threshold(importances, threshold) == expected_result

def test_check_threshold_string_unknown_reference(importances):
    threshold = "3 * max"
    with pytest.raises(ValueError):
        _check_threshold(importances, threshold)

def test_check_threshold_float(importances):
    threshold = 0.4
    expected_result = 0.4
    assert _check_threshold(importances, threshold) == expected_result