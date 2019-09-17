from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.externals import six

from collections import defaultdict

from sklearn.pipeline import _name_estimators
from imblearn import pipeline


__all__ = ['Pipeline', 'make_pipeline']



class Pipeline(pipeline.Pipeline):

    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_

    @property
    def coef_(self):
        return self._final_estimator.coef_





def make_pipeline(*steps, **kwargs):
    """Construct a Pipeline from the given estimators.

    This is a shorthand for the Pipeline constructor; it does not require, and
    does not permit, naming the estimators. Instead, their names will be set
    to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of estimators.

    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    Returns
    -------
    p : Pipeline

    """
    memory = kwargs.pop('memory', None)
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    return Pipeline(_name_estimators(steps), memory=memory)
