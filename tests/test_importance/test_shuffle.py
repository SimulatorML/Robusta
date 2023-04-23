import pandas as pd
import pytest

from robusta.importance import _shuffle_data


@pytest.fixture
def data():
    x = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    return x, y


def test_shuffle_data(data):
    # Ensure that the function shuffles the data correctly
    x, y = data
    x_shuffled, y_shuffled = _shuffle_data(x, y, seed=42)

    assert x_shuffled.equals(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}))
    assert y_shuffled.equals(pd.Series([0, 1, 0]))

    # Ensure that the function returns the same indexes for all input data
    assert all([x_shuffled.index.equals(x.index), y_shuffled.index.equals(y.index)])

    # Ensure that the function does not modify input data
    assert x.equals(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}))
    assert y.equals(pd.Series([0, 1, 0]))

    # Ensure that the function handles None input data
    z = None
    x_shuffled, y_shuffled, z_shuffled = _shuffle_data(x, y, z, seed=42)
    assert z_shuffled is None
