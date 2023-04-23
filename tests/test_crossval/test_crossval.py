import pytest
from sklearn.linear_model import LinearRegression

from robusta.crossval import copy


@pytest.fixture
def estimator():
    return LinearRegression()


def test_copy(estimator):
    # Make a copy of the estimator using the copy function
    estimator_copy = copy(estimator)

    # Check that the copy is not the same object as the original
    assert estimator_copy is not estimator

    # Check that the copy has the same type as the original
    assert type(estimator_copy) == type(estimator)

    # Check that the copy has the same parameters as the original
    assert estimator_copy.get_params() == estimator.get_params()
