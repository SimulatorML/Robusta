import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from robusta.preprocessing import FastEncoder, NaiveBayesEncoder


def test_fast_encoder():
    # Generate some sample data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=['col1', 'col2', 'col3', 'col4', 'col5'])
    X['col6'] = pd.Series(np.random.choice(['a', 'b', 'c'], size=len(X)))
    y = pd.Series(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the encoder
    encoder = FastEncoder(smoothing=1.0, min_samples_leaf=1)
    encoder.fit(X_train, y_train)

    # Transform the training data
    X_train_enc = encoder.transform(X_train)

    # Ensure that the transformed data has the same number of rows and columns
    assert X_train_enc.shape == X_train.shape

    # Ensure that the transformed data is not equal to the original data
    assert not np.array_equal(X_train_enc.values, X_train.values)

    # Transform the testing data
    X_test_enc = encoder.transform(X_test)

    # Ensure that the transformed testing data has the same number of columns as the training data
    assert X_test_enc.shape[1] == X_train_enc.shape[1]


def test_NaiveBayesEncoder():
    # custom data
    X = csr_matrix(np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]]))
    y = np.array([0, 1, 0])

    # initialize and fit the encoder
    encoder = NaiveBayesEncoder(smooth=1.0)
    encoder.fit(X, y)

    # transform the input data
    X_transformed = encoder.transform(X)
    print(type(X_transformed))

    # check that the shape of the output is correct
    assert X_transformed.shape == X.shape
    print(X_transformed.toarray().tolist())
    # check that the output values are correct
    assert np.allclose(X_transformed.toarray(),
                       np.array([[0.0, 0.4054651081081644, 0.0],
                                 [1.0986122886681098, 0.4054651081081644, 0.4054651081081644],
                                 [0.0, 0.0, 0.4054651081081644]]))