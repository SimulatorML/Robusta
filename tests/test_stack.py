import pandas as pd

from robusta.stack import stack_preds


def test_stack_preds():
    pred_list = [
        pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
        pd.Series([5, 6], name='C'),
    ]
    names = ['foo', 'bar']

    expected_output = pd.DataFrame({
        ('A', 'foo'): [1, 2],
        ('B', 'foo'): [3, 4],
        'bar': [5, 6],
    })

    output = stack_preds(pred_list, names)
    pd.testing.assert_frame_equal(output, expected_output)