import pandas as pd
import pytest

from robusta.optimizer import fix_params, ranking, qround

############################################################################################
#####################################FixParams##############################################
############################################################################################

def test_fix_params():
    # Define test parameters
    params = {'param1': 3.1415, 'param2': 10, 'param3': 'hello'}
    space = {'param1': (0, 10, 0.1), 'param2': (0, 20), 'param3': 'world'}

    # Call the function to fix the parameters
    fixed_params = fix_params(params, space)

    # Check that the parameters were correctly fixed
    assert fixed_params['param1'] == pytest.approx(3.1, 0.1)
    assert fixed_params['param2'] == 10
    assert fixed_params['param3'] == 'world'

############################################################################################
#######################################Ranking##############################################
############################################################################################

def test_ranking():
    # Define test series
    ser = pd.Series([3, 1, 2, 5, 4, None])

    # Call the function to rank the values
    rnk = ranking(ser)

    # Check that the output is correct
    expected_rnk = pd.Series([3, 5, 4, 1, 2, 5])
    assert rnk.equals(expected_rnk)

############################################################################################
########################################QRound##############################################
############################################################################################

def test_qround():
    # Test with integer values
    assert qround(3, 0, 10, 2) == 2
    assert qround(7, 0, 10, 2) == 6
    assert qround(11, 0, 10, 2) == 10
    assert qround(5, 0, 10, 2, decimals=0) == 4
    assert qround(3.3, 0, 10, 2) == 2
    assert qround(7.7, 0, 10, 2) == 6.0
    assert qround(11.1, 0, 10, 2) == 10.0
    assert qround(5.5, 0, 10, 2, decimals=0) == 4.0

    # Test with float values
    assert qround(0.35, 0.0, 1.0, 0.1) == 0.3
    assert qround(0.55, 0.0, 1.0, 0.1) == 0.5
    assert qround(1.1, 0.0, 1.0, 0.1) == 0.9
    assert qround(0.5, 0.0, 1.0, 0.1, decimals=0) == 0.0
    assert qround(0.35, 0.0, 1.0, 0.1, decimals=2) == 0.30
    assert qround(0.55, 0.0, 1.0, 0.1, decimals=2) == 0.50
    assert qround(1.1, 0.0, 1.0, 0.1, decimals=2) == 0.9
    assert qround(0.5, 0.0, 1.0, 0.1, decimals=0) == 0.0

    # Test with edge cases
    assert qround(5, 0, 10, 1) == 5
    assert qround(5, 0, 10, 10) == 0
    assert qround(-1, 0, 10, 2) == 0
    assert qround(11, 0, 10, 2) == 10
    assert qround(3, 0, 10, -1) == 3

    with pytest.raises(ZeroDivisionError):
        qround(5, 0, 10, 0)
