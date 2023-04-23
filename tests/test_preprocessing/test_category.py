import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from robusta.preprocessing import Categorizer
from robusta.preprocessing import Categorizer1D
from robusta.preprocessing import FrequencyEncoder
from robusta.preprocessing import LabelBinarizer
from robusta.preprocessing import LabelEncoder
from robusta.preprocessing import LabelEncoder1D, SVDEncoder, ThermometerEncoder1D, ThermometerEncoder, GroupByEncoder

############################################################################################
#######################################LabelEncoder1D#######################################
############################################################################################

def test_label_encoder_1d():
    # Load iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the preprocessing pipeline
    preprocess = LabelEncoder1D()

    # Fit and transform the training data
    preprocess.fit(y_train)
    y_train_transformed = preprocess.transform(y_train)

    # Check that the number of unique categories is correct
    assert len(np.unique(y_train)) == len(np.unique(y_train_transformed))

    # Check that the inverse transform works correctly
    y_train_inverse_transformed = preprocess.inverse_transform(y_train_transformed)
    assert y_train.equals(y_train_inverse_transformed)

############################################################################################
#########################################LabelEncoder#######################################
############################################################################################

def test_LabelEncoder():
    # create test data
    X = pd.DataFrame({'color': ['red', 'green', 'blue', 'red'],
                      'size': ['small', 'medium', 'large', 'medium']})

    # create instance of LabelEncoder
    le = LabelEncoder()

    # fit the encoder to the data
    le.fit(X)

    # transform the data
    X_transformed = le.transform(X)

    # check that the transformed data has the correct shape and data type
    assert X_transformed.shape == X.shape
    assert isinstance(X_transformed, pd.DataFrame)
    assert np.issubdtype(X_transformed['color'].dtype, np.integer)
    assert np.issubdtype(X_transformed['size'].dtype, np.integer)

    # inverse transform the data
    X_inverse = le.inverse_transform(X_transformed)

    # check that the inverse transformed data is the same as the original data
    assert X.equals(X_inverse)

############################################################################################
########################################Categorizer1D#######################################
############################################################################################

@pytest.fixture
def data():
    return pd.Series(['cat', 'dog', 'bird', 'cat', 'dog', 'bird'])


def test_categorizer_transform(data):
    categorizer = Categorizer1D()
    categorizer.fit(data)
    transformed_data = categorizer.transform(data)
    assert pd.api.types.is_categorical_dtype(transformed_data.dtype)
    assert all(transformed_data.categories == categorizer.cats_)
    assert len(transformed_data) == len(data)

############################################################################################
##########################################FrequencyEncoder##################################
############################################################################################

@pytest.fixture
def sample_data():
    data = {
        'col1': ['A', 'A', 'B', 'C'],
        'col2': [1, 2, 2, 3],
        'col3': [True, False, True, False],
    }
    return pd.DataFrame(data)


def test_frequency_encoder(sample_data):
    encoder = FrequencyEncoder()
    encoder.fit(sample_data)
    encoded_data = encoder.transform(sample_data)

    assert encoded_data.shape == sample_data.shape
    assert encoded_data.dtypes.unique() == float


############################################################################################
############################################SVDEncoder######################################
############################################################################################

@pytest.fixture
def data():
    return pd.DataFrame({
        'cat1': ['a', 'a', 'b', 'b', 'c', 'c'],
        'cat2': ['x', 'y', 'y', 'z', 'z', 'x'],
        'target': [0, 1, 0, 1, 0, 1]
    })

def test_missing_values(data):
    encoder = SVDEncoder()
    with pytest.raises(AssertionError):
        encoder.fit_transform(pd.DataFrame({'cat1': ['a', 'b', pd.NA], 'cat2': ['x', 'y', 'z']}))


def test_unknown_n_components():
    with pytest.raises(ValueError):
        encoder = SVDEncoder(n_components='invalid_type')
        encoder.fit(pd.DataFrame({'cat1': ['a', 'a'], 'cat2': ['x', 'y']}))


def test_n_components_int():
    encoder = SVDEncoder(n_components=2)
    assert encoder.n_components == 2


def test_n_components_float():
    encoder = SVDEncoder(n_components=0.9)
    assert encoder.n_components == 0.9

############################################################################################
########################################LabelBinarizer######################################
############################################################################################

def test_LabelBinarizer_fit_transform():
    # test binary target variable
    y = pd.Series([0, 1, 1, 0, 1])
    lb = LabelBinarizer().fit(y)
    assert lb.classes_.tolist() == [0, 1]
    y_transformed = lb.transform(y)
    assert y_transformed.tolist() == [0, 1, 1, 0, 1]
    y_inverse_transformed = lb.inverse_transform(y_transformed)
    assert y_inverse_transformed.tolist() == y.tolist()

    # test invalid target variable
    y = pd.Series([])
    with pytest.raises(ValueError):
        lb = LabelBinarizer().fit(y)

############################################################################################
########################################ThermometerEncoder1D################################
############################################################################################

@pytest.fixture
def data():
    return pd.Series(['a', 'b', 'c', 'a', 'c', 'b', 'd', 'a'])


@pytest.fixture
def encoder():
    return ThermometerEncoder1D()


def test_inverse_transform(encoder, data):
    encoded_data = encoder.fit_transform(data)
    decoded_data = encoder.inverse_transform(encoded_data)
    assert isinstance(decoded_data, pd.Series)
    assert decoded_data.dtype == encoder.type_
    assert decoded_data.name == encoder.name_
    assert decoded_data.eq(data).all()


############################################################################################
########################################ThermometerEncoder##################################
############################################################################################

def test_transform():
    # create a sample dataframe
    df = pd.DataFrame({
        'col1': [0, 1, 2, 3],
        'col2': [1, 2, 1, 2]
    })

    # create a transformer object
    encoder = ThermometerEncoder()

    # fit and transform the dataframe
    encoder.fit(df)
    encoded_df = encoder.transform(df)

    # check the output
    expected_output = pd.DataFrame({
        'col1:0': [1, 1, 1, 1],
        'col1:1': [0, 1, 1, 1],
        'col1:2': [0, 0, 1, 1],
        'col1:3': [0, 0, 0, 1],
        'col2:1': [1, 1, 1, 1],
        'col2:2': [0, 1, 0, 1]
    })
    assert (encoded_df == expected_output).all().all()

############################################################################################
##########################################GroupByEncoder####################################
############################################################################################

def test_GroupByEncoder():
    # Create sample data
    X = pd.DataFrame({
        'A': ['a', 'b', 'a', 'b'],
        'B': [1, 2, 3, 4],
        'C': [10, 20, 30, 40]
    })

    # Initialize transformer
    encoder = GroupByEncoder()

    # Test fit method
    encoder.fit(X)
    assert encoder.cats_ == ['A']
    assert encoder.nums_ == ['B', 'C']

    # Test transform method
    Xt = encoder.transform(X)
    assert 'B__A' in Xt.columns
    assert 'C__A' in Xt.columns
    assert Xt.shape == (4, 2)

    # Test transform method with diff=True
    encoder = GroupByEncoder(func='mean', diff=True)
    encoder.fit(X)
    Xt = encoder.transform(X)
    assert 'B__A' in Xt.columns
    assert 'C__A' in Xt.columns

    assert Xt['B__A'].values.tolist() == [-1.0, -1.0, 1.0, 1.0]
    assert Xt['C__A'].values.tolist() == [-10, -10, 10, 10]
