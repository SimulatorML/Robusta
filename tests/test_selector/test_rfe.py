import pandas as pd
import pytest

from robusta.selector import _check_step, _select_k_best

############################################################################################
#####################################CheckStep##############################################
############################################################################################

def test_check_step():
    # Test integer step parameter
    assert _check_step(2, 10, 5) == 2

    # Test integer step parameter less than or equal to 0
    with pytest.raises(ValueError, match='Integer <step> must be greater than 0'):
        _check_step(0, 10, 5)

    # Test float step parameter within range
    assert _check_step(0.5, 10, 5) == 5

    # Test step larger than difference between total number of features and desired number of features
    assert _check_step(10, 5, 2) == 3

############################################################################################
###################################SelectKBest##############################################
############################################################################################

def test_select_k_best():
    # Create a sample Pandas Series of feature scores
    feature_scores = pd.Series([0.5, 0.2, 0.9, 0.7, 0.1])

    # Test selecting the 3 best features
    assert _select_k_best(feature_scores, 3).tolist() == [2, 3, 0]

    # Test selecting the 1 best feature
    assert _select_k_best(feature_scores, 1).tolist() == [2]

    # Test selecting all features (should return the same order as the original series)
    assert _select_k_best(feature_scores, 5).tolist() == [2, 3, 0, 1, 4]

    # Test selecting 0 features (should return an empty list)
    assert _select_k_best(feature_scores, 0).tolist() == []
