import pytest

from robusta.selector import _check_k_features


def test_check_k_features():
    # Test valid integer k_features
    assert _check_k_features(10, 100) == 10

    # Test invalid integer k_features
    with pytest.raises(ValueError):
        _check_k_features(0, 100)

    # Test valid float k_features
    assert _check_k_features(0.1, 100) == 10

    # Test invalid float k_features
    with pytest.raises(ValueError):
        _check_k_features(1.5, 100)

    # Test invalid parameter type
    with pytest.raises(ValueError):
        _check_k_features('a', 100)