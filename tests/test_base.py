import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import ColumnRenamer
from preprocessing import ColumnSelector
from preprocessing import Identity
from preprocessing import SimpleImputer
from preprocessing import TypeConverter
from preprocessing import TypeSelector
from preprocessing import ColumnGrouper
from preprocessing import FunctionTransformer


############################################################################################
################################ColumnSelector##############################################
############################################################################################
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c'],
        'C': [4.0, 5.0, 6.0]
    })

def test_transform_with_iterable_columns(sample_data):
    transformer = ColumnSelector(columns=['A', 'C'])
    transformed_data = transformer.transform(sample_data)
    assert transformed_data.equals(sample_data[['A', 'C']])


def test_transform_with_missing_columns(sample_data):
    transformer = ColumnSelector(columns=['A', 'D'])
    with pytest.raises(KeyError):
        transformer.transform(sample_data)


############################################################################################
##################################TypeSelector##############################################
############################################################################################

def test_typeselector():
    iris = load_iris()
    X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    y = pd.Series(data=iris.target, name="target")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Define pipeline with TypeSelector
    pipe = TypeSelector(dtype=np.number)


    # Fit and transform pipeline on training data
    Xt_train = pipe.fit_transform(X_train)

    # Test that transformed data only contains numeric columns
    assert set(Xt_train.columns) == set(X_train.select_dtypes(include=[np.number]).columns)

    # Transform testing data using the fitted pipeline
    Xt_test = pipe.transform(X_test)

    # Test that transformed testing data only contains numeric columns
    assert set(Xt_test.columns) == set(X_test.select_dtypes(include=[np.number]).columns)


@pytest.fixture
def data():
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c'],
        'col3': [True, False, True]
    })
    return df


############################################################################################
#################################TypeConverter##############################################
############################################################################################

def test_TypeConverter(data):
    # Test fit method
    tc = TypeConverter(dtypes={'col1': float, 'col2': str, 'col3': bool})
    tc.fit(data)
    assert tc.dtypes_old_.equals(data.dtypes)
    assert tc.dtypes == {'col1': float, 'col2': str, 'col3': bool}

    # Test transform method
    transformed = tc.transform(data)
    expected_dtypes = pd.Series(['float64', 'object', 'bool'], index=['col1', 'col2', 'col3'])
    assert transformed.dtypes.equals(expected_dtypes)

    # Test inverse_transform method
    inverse_transformed = tc.inverse_transform(transformed)
    assert inverse_transformed.dtypes.equals(pd.Series(tc.dtypes_old_))
    assert inverse_transformed.equals(data)


############################################################################################
######################################Identity##############################################
############################################################################################

def test_identity_transformer():
    identity = Identity()
    X = np.array([[1, 2], [3, 4]])
    assert np.array_equal(identity.transform(X), X)


def test_identity_fit():
    identity = Identity()
    assert identity.fit() is identity


############################################################################################
######################################ColumnRenamer#########################################
############################################################################################

def test_column_renamer():
    # create a sample DataFrame
    X = pd.DataFrame({
        'a': np.random.randn(5),
        'b': np.random.randn(5),
        'c': np.random.randn(5)
    })
    # create a ColumnRenamer instance
    renamer = ColumnRenamer(column=['a', 'b', 'c'], prefix='new_', suffix='_col')

    renamer.fit(X)
    Xt = renamer.transform(X)

    assert list(Xt.columns) == ['new_a_col', 'new_b_col', 'new_c_col']


############################################################################################
######################################SimpleImputer#########################################
############################################################################################

def test_simple_imputer():
    # Load iris dataset and add missing values
    iris = load_iris()
    X, y = iris['data'], iris['target']
    X_with_missing = pd.DataFrame(X).replace(0, np.nan)

    # Test mean strategy
    imp = SimpleImputer(strategy='mean')
    imp.fit(X_with_missing)
    X_imputed = imp.transform(X_with_missing)
    assert np.isnan(X_imputed).sum().all() == 0

    # Test median strategy
    imp = SimpleImputer(strategy='median')
    imp.fit(X_with_missing)
    X_imputed = imp.transform(X_with_missing)
    assert np.isnan(X_imputed).sum().all() == 0

    # Test mode strategy
    imp = SimpleImputer(strategy='mode')
    imp.fit(X_with_missing)
    X_imputed = imp.transform(X_with_missing)
    assert np.isnan(X_imputed).sum().all() == 0

    # Test const strategy
    imp = SimpleImputer(strategy='const', fill_value=1)
    imp.fit(X_with_missing)
    X_imputed = imp.transform(X_with_missing)
    assert np.isnan(X_imputed).sum().all() == 0

    # Test non-numeric columns
    df = pd.DataFrame({'A': ['foo', 'bar'], 'B': [1, np.nan]})
    imp = SimpleImputer(strategy='mean')
    with pytest.raises(ValueError):
        imp.fit(df)


############################################################################################
######################################ColumnGrouper#########################################
############################################################################################

def test_ColumnGrouper():
    # create test data
    X = pd.DataFrame({'A': np.random.randn(5),
                      'B': np.random.randn(5),
                      'C': np.random.randn(5)})

    # initialize the transformer
    transformer = ColumnGrouper(group='Group1')

    # check that the output has expected columns
    expected_columns = pd.MultiIndex.from_arrays([['Group1'] * len(X.columns), X.columns])
    assert transformer.fit_transform(X).columns.equals(expected_columns)


def test_function_transformer():
    # create sample data
    X = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    # create function transformer
    func = lambda x: np.log(x)
    inverse_func = lambda x: np.exp(x)
    transformer = FunctionTransformer(func=func, inverse_func=inverse_func)

    # test transform method
    transformed_X = transformer.transform(X)
    expected_transformed_X = pd.DataFrame({'A': [0.0, 0.6931471805599453, 1.0986122886681098],
                                           'B': [1.3862943611198906, 1.6094379124341003, 1.791759469228055]})
    assert transformed_X.equals(expected_transformed_X)

    # test inverse_transform method
    inverse_transformed_X = transformer.inverse_transform(transformed_X)
    expected_inverse_transformed_X = pd.DataFrame({'A': [1., 2., 3.0000000000000004],
                                                   'B': [4., 4.999999999999999, 6.]})
    assert inverse_transformed_X.equals(expected_inverse_transformed_X)
