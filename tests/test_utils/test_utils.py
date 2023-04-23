import numpy as np
import pytest

from robusta.utils import all_subsets, get_ranks, bytefmt, sizeof, ld2dl

############################################################################################
####################################AllSubsets##############################################
############################################################################################

def test_all_subsets():
    # Test case 1
    cols1 = ['a', 'b', 'c']
    k_range1 = (1, 2)
    expected_output1 = [('a',), ('b',), ('c',), ('a', 'b'), ('a', 'c'), ('b', 'c')]
    assert list(all_subsets(cols1, k_range1)) == expected_output1

    # Test case 2
    cols2 = [1, 2, 3, 4]
    k_range2 = (2, 3)
    expected_output2 = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]
    assert list(all_subsets(cols2, k_range2)) == expected_output2

    # Test case 3
    cols3 = ['hello', 'world']
    k_range3 = (0, 1)
    expected_output3 = [(), ('hello',), ('world',)]
    assert list(all_subsets(cols3, k_range3)) == expected_output3

    # Test case 4
    cols4 = []
    k_range4 = (1, 3)
    expected_output4 = []
    assert list(all_subsets(cols4, k_range4)) == expected_output4

    # Test case 5
    cols5 = ['x', 'y', 'z']
    k_range5 = (3, 5)
    expected_output5 = [('x', 'y', 'z')]
    assert list(all_subsets(cols5, k_range5)) == expected_output5

    # Test case 6
    cols6 = ['a']
    k_range6 = (1, 5)
    expected_output6 = [('a',)]
    assert list(all_subsets(cols6, k_range6)) == expected_output6

    # Test case 7
    cols7 = [1, 2, 3]
    k_range7 = (0, 0)
    expected_output7 = [(), ()]
    assert list(all_subsets(cols7, k_range7)) == expected_output7

    # Test case 8
    cols8 = ['x', 'y']
    k_range8 = (0, 0)
    expected_output8 = [(), ()]
    assert list(all_subsets(cols8, k_range8)) == expected_output8

############################################################################################
######################################GetRanks##############################################
############################################################################################

def test_get_ranks():
    arr = [3.5, 1.2, 5.1, 2.4]
    expected_ranks = [2, 0, 3, 1]
    assert np.array_equal(get_ranks(arr), expected_ranks)

def test_get_ranks_normalize():
    arr = [3.5, 1.2, 5.1, 2.4]
    expected_ranks = [0.33333333, 0.0, 0.5, 0.16666667]
    assert np.allclose(get_ranks(arr, normalize=True), expected_ranks, rtol=1e-5)

def test_get_ranks_empty_array():
    arr = []
    expected_ranks = []
    assert np.array_equal(get_ranks(arr), expected_ranks)

def test_get_ranks_array_with_one_element():
    arr = [2.3]
    expected_ranks = [0]
    assert np.array_equal(get_ranks(arr), expected_ranks)

############################################################################################
#######################################Bytefmt##############################################
############################################################################################

def test_bytefmt():
    # Test with exact byte representation
    assert bytefmt(1) == '1 bytes'
    assert bytefmt(1023) == '1023 bytes'
    assert bytefmt(1024) == '1 KB'
    assert bytefmt(1025) == '1 KB'
    assert bytefmt(1048575) == '1024 KB'
    assert bytefmt(1048576) == '1 MB'
    assert bytefmt(1073741823) == '1024 MB'
    assert bytefmt(1073741824) == '1 GB'
    assert bytefmt(1099511627775) == '1024 GB'
    assert bytefmt(1099511627776) == '1 TB'

    # Test with rounded byte representation
    assert bytefmt(0, rnd=True) == ''
    assert bytefmt(1023, rnd=True) == '1023 bytes'
    assert bytefmt(1024, rnd=True) == '1 KB'
    assert bytefmt(1025, rnd=True) == '1 KB'
    assert bytefmt(1048575, rnd=True) == '1024 KB'
    assert bytefmt(1048576, rnd=True) == '1 MB'
    assert bytefmt(1073741823, rnd=True) == '1024 MB'
    assert bytefmt(1073741824, rnd=True) == '1 GB'
    assert bytefmt(1099511627775, rnd=True) == '1024 GB'
    assert bytefmt(1099511627776, rnd=True) == '1 TB'

############################################################################################
#######################################SizeOf###############################################
############################################################################################

def test_sizeof():
    # Test with a small object
    obj = [1, 2, 3]
    assert sizeof(obj) == '184 bytes'  # The exact size may vary depending on the system.

    # Test with a larger object
    obj = [i for i in range(1000000)]
    assert sizeof(obj) == '38.6 MB'  # The exact size may vary depending on the system.

    # Test with formatting enabled
    obj = [i for i in range(1000000)]
    assert sizeof(obj, fmt=True) == '38.6 MB'

    # Test with rounding disabled
    obj = [i for i in range(1000000)]
    assert sizeof(obj, fmt=True, rnd=False) == '38 MB  588 KB  720 bytes'

############################################################################################
#######################################Ld2dl################################################
############################################################################################

def test_ld2dl():
    # Test case 1: single dictionary input
    input1 = [{"a": 1, "b": 2}]
    expected_output1 = {"a": [1], "b": [2]}
    assert ld2dl(input1) == expected_output1

    # Test case 2: multiple dictionaries with same keys input
    input2 = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]
    expected_output2 = {"a": [1, 3, 5], "b": [2, 4, 6]}
    assert ld2dl(input2) == expected_output2

    # Test case 5: input with one dictionary having missing key
    input4 = [{"a": 1, "b": 2}, {"a": 3}]
    with pytest.raises(KeyError):
        ld2dl(input4)