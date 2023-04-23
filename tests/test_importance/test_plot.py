import pandas as pd
from matplotlib import pyplot as plt

from robusta.importance import plot_importance


def test_plot_importance():
    # Define test data
    imps = [pd.Series([0.1, 0.2, 0.3, 0.4], index=['a', 'b', 'c', 'd']),
            pd.Series([0.2, 0.3, 0.1, 0.4], index=['a', 'b', 'c', 'd']),
            pd.Series([0.3, 0.1, 0.2, 0.4], index=['a', 'b', 'c', 'd'])]
    features = ['a', 'b', 'c', 'd']
    k_top = 3

    # Call function
    plot_importance(imps, features, k_top)

    # Check that the plot was created successfully
    assert plt.gcf().number == 1, 'The plot was not created successfully'