import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, load_iris
from sklearn.preprocessing import normalize

from robusta.preprocessing import KBinsDiscretizer1D
from robusta.preprocessing import MaxAbsScaler
from robusta.preprocessing import RobustScaler
from robusta.preprocessing import StandardScaler
from robusta.preprocessing import SyntheticFeatures, MinMaxScaler, Normalizer, Binarizer

############################################################################################
#########################################SyntheticFeatures##################################
############################################################################################

@pytest.fixture
def example_input():
    # Generate an example input dataframe
    return pd.DataFrame({'x_0': [1, 2, 3], 'x_1': [4, 5, 6]})


def test_fit(example_input):
    # Test that the fit method initializes the columns attribute correctly
    transformer = SyntheticFeatures()
    transformer.fit(example_input)

    assert (transformer.columns == ['x_0', 'x_1']).all()

############################################################################################
###########################################RobustScaler#####################################
############################################################################################

@pytest.fixture
def example_data():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X_df


def test_RobustScaler(example_data):
    scaler = RobustScaler()
    X_transformed = scaler.fit_transform(example_data)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape == example_data.shape

    # Define column names and values
    column_names = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
    column_values = [-0.0848188467127898, 0.0625359292926655, -0.006378779861593824,
                     -0.04288798709087402, -0.019312139269701788]

    # Create DataFrame using Pandas
    df = pd.Series(column_values, index=column_names)

    # Check centering
    if scaler.centering:
        assert (X_transformed.mean() == df).all()

############################################################################################
###########################################StandardScaler###################################
############################################################################################

def test_standard_scaler():
    # Generate random data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    scalar = StandardScaler()

    # Fit and transform the data
    X_transformed = scalar.fit_transform(X)

    mean_list = [0.04774065005627248, -0.033572656952548525, -0.013860667063059808, -0.017117658763571318,
                 -0.005800134649109627, 0.03420700308360378, 0.05511867072212924, -0.10372591796925082,
                 -0.11936617587463123, 0.15637688741016573]
    std_list = [1.1826510107803108, 1.141586341064498, 1.2286635296928805, 0.8305706324045815,
                0.8514852206309808, 0.9550304156823306, 0.9192424960914402, 0.8863053092573022,
                1.0030134365947512, 0.8759972806057073]

    # Check that the mean is zero and the standard deviation is one
    assert np.allclose(X_transformed.mean(axis=0), mean_list)
    assert np.allclose(X_transformed.std(axis=0), std_list)

############################################################################################
###########################################MinMaxScaler#####################################
############################################################################################

@pytest.fixture
def iris_data():
    return load_iris(return_X_y=True)[0]


def test_minmax_scaler(iris_data):
    pipeline = MinMaxScaler()

    X_scaled = pipeline.fit_transform(iris_data)
    mean_list = [0.7363247863247862, 0.3791452991452987, 0.4689743589743589, 0.1409401709401709]
    std_list = [0.1058078579211719, 0.055693713812242925, 0.22556462381734665, 0.09739649075668703]

    assert np.allclose(X_scaled.mean(axis=0), mean_list, atol=1e-6)
    assert np.allclose(X_scaled.std(axis=0), std_list, atol=1e-6)

############################################################################################
###########################################MaxAbsScaler#####################################
############################################################################################

def test_MaxAbsScaler():
    # Create DataFrame
    data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
    df = pd.DataFrame(data)

    # Create MaxAbsScaler object and fit to DataFrame
    scaler = MaxAbsScaler()
    scaler.fit(df)

    # Check that scaling factors are calculated correctly
    expected_scale = pd.Series([3, 6, 9], index=['A', 'B', 'C'])
    pd.testing.assert_series_equal(scaler.scale_, expected_scale, check_names=False)

    # Check that transform works correctly
    transformed = scaler.transform(df)
    expected_transformed = pd.DataFrame({'A': [1 / 3, 2 / 3, 1], 'B': [4 / 6, 5 / 6, 1], 'C': [7 / 9, 8 / 9, 1]})
    pd.testing.assert_frame_equal(transformed, expected_transformed)

############################################################################################
###########################################Normalizer#######################################
############################################################################################

def test_Normalizer_transform():
    # Create a sample dataset
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    columns = ['A', 'B', 'C']
    index = ['Row1', 'Row2', 'Row3']
    df = pd.DataFrame(X, columns=columns, index=index)

    # Initialize Normalizer object
    normalizer = Normalizer()

    # Test transform() method
    transformed_X = normalizer.transform(X)
    transformed_df = normalizer.transform(df)

    # Check that output is a numpy array
    assert isinstance(transformed_X, np.ndarray)
    assert isinstance(transformed_df, pd.DataFrame)

    # Check that output has the same shape as input
    assert transformed_X.shape == X.shape
    assert transformed_df.shape == df.shape

    # Check that output is normalized
    expected_X = normalize(X, axis=1)
    expected_df = pd.DataFrame(expected_X, columns=columns, index=index)
    np.testing.assert_array_almost_equal(transformed_X, expected_X)
    pd.testing.assert_frame_equal(transformed_df, expected_df)


############################################################################################
###########################################KBinsDiscretizer1D###############################
############################################################################################

def test_k_bins_discretizer_1d():
    # Generate some random data
    np.random.seed(42)
    y = np.random.rand(1000)

    # Instantiate the transformer with different strategies and number of bins
    for strategy in ['quantile', 'uniform']:
        for bins in [3, 5]:
            transformer = KBinsDiscretizer1D(bins=bins, strategy=strategy)

            # Test the fit and transform methods
            transformed = transformer.fit_transform(y)
            assert len(transformed) == len(y)

############################################################################################
#############################################Binarizer######################################
############################################################################################

def test_binarizer_transform():
    # Create a sample dataset
    X = np.array([[1, 2, 3], [4, 5, 6]])

    # Initialize the transformer
    binarizer = Binarizer(threshold=3)

    # Transform the data
    X_transformed = binarizer.transform(X)

    # Test the output shape and data type
    assert X_transformed.shape == (2, 3)
    assert X_transformed.dtype == np.uint8

    # Test that the transformer applied the threshold correctly
    expected_output = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(X_transformed, expected_output)