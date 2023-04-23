import numpy as np
import pandas as pd

from robusta.preprocessing import CyclicEncoder
from robusta.preprocessing import DatetimeConverter1D, DatetimeConverter


############################################################################################
#######################################DatetimeConverter1D##################################
############################################################################################

class TestDatetimeConverter1D:
    def setup_method(self):
        self.transformer = DatetimeConverter1D(format='%Y-%m-%d %H:%M:%S')

    def test_transform(self):
        input_data = pd.Series(['2022-01-01 12:00:00', '2022-01-02 12:00:00'])
        expected_output = pd.to_datetime(input_data, format='%Y-%m-%d %H:%M:%S')
        assert self.transformer.transform(input_data).equals(expected_output)

############################################################################################
#######################################DatetimeConverter####################################
############################################################################################

def test_datetime_converter():
    # Create a toy dataset with some date columns
    dates = pd.date_range(start='2020-01-01', end='2020-01-05')
    data = {'date1': dates,
            'date2': dates + pd.Timedelta(days=1),
            'numeric_col': np.random.randn(len(dates))}
    df = pd.DataFrame(data)

    # Create an instance of the transformer
    transformer = DatetimeConverter()

    # Test that the transformer converts the date columns to datetime objects
    transformed = transformer.transform(df)
    assert isinstance(transformed['date1'][0], pd.Timestamp)
    assert isinstance(transformed['date2'][0], pd.Timestamp)

############################################################################################
#########################################CyclicEncoder######################################
############################################################################################

def test_cyclic_encoder():
    # Generate some test data
    X = pd.DataFrame({'x1': [0, 1, 2, 3, 4], 'x2': [10, 20, 30, 40, 50]})
    encoder = CyclicEncoder(delta=1)
    encoder.fit(X)

    # Check if the output shape is correct
    transformed = encoder.transform(X)
    assert transformed.shape == (5, 4)

    # Check if the output contains the expected columns
    expected_cols = ['x1_cos', 'x1_sin', 'x2_cos', 'x2_sin']
    assert list(transformed.columns) == expected_cols

    # Check if the output is within the expected range
    assert np.all(transformed >= -1) and np.all(transformed <= 1)

    # Check if the output has the correct values
    expected = pd.DataFrame({
        'x1_cos': [1, 0.980067, 0.921061, 0.825336, 0.696707],
        'x1_sin': [0, 0.198669, 0.389418, 0.564642, 0.717356],
        'x2_cos': [1, 0.970403, 0.883364, 0.744035, 0.560663],
        'x2_sin': [0, 0.241491, 0.468688, 0.668141, 0.828044]
    })

    np.testing.assert_allclose(transformed, expected, rtol=1e-3)
