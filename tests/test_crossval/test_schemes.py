import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from robusta.crossval import shuffle_labels, make_adversarial_validation, AdversarialValidation


def test_shuffle_labels():
    # Create a sample input Series with unique values
    input_series = pd.Series([1, 2, 3, 4, 5])

    # Shuffle the input Series using shuffle_labels
    shuffled_series = shuffle_labels(input_series, random_state=42)

    # Check that the shuffled series has the same unique values as the input series
    assert set(shuffled_series.unique()) == set(input_series.unique())

    # Check that the shuffled series is not equal to the input series
    assert not shuffled_series.equals(input_series)

    # Check that shuffling the same input with the same random_state produces the same output
    assert shuffle_labels(input_series, random_state=42).equals(shuffled_series)

    # Check that shuffling a different input with the same random_state produces a different output
    assert not shuffle_labels(pd.Series([6, 7, 8, 9, 10]), random_state=42).equals(shuffled_series)

    # Check that shuffling an input with non-unique values raises a ValueError
    with pytest.raises(ValueError):
        shuffle_labels(pd.Series([1, 2, 3, 3, 4]))

    # Check that passing a non-pandas Series input raises a TypeError
    with pytest.raises(TypeError):
        shuffle_labels([1, 2, 3, 4, 5], random_state=42)
