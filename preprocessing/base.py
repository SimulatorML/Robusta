import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn import impute

from typing import Iterable


__all__ = [
    'TypeSelector',
    'ColumnSelector',
    'ColumnRenamer',
    'Imputer',
    'Identity',
]




class PandasTransformer(BaseEstimator, TransformerMixin):
    '''Wrapper for sklearn transformers, that takes and returns pandas DataFrames.

    Parameters
    ----------
    transformer : estimator
        Core transformer

    **params :
        Set the parameters of core estimator.

    '''
    def __init__(self, transformer, **params):
        self.transformer = transformer.set_params(**params)


    def fit(self, X, y=None):
        """Fit core transformer using X.

        Parameters
        ----------
        X : DataFrame of shape [n_samples, n_features]
            The data to fit.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self : ColumnTransformer
            This estimator

        """
        self.transformer.fit(X, y)
        self.columns = list(X.columns)

        return self


    def transform(self, X):
        """Transform X using specified transformer.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        Xt : DataFrame, shape [n_samples, n_features]
            Transformed input.

        """
        Xt = self.transformer.transform(X)
        Xt = pd.DataFrame(Xt, columns=self.columns, index=X.index)

        return Xt


    #def set_params(self, **params):
    #    self.transformer.set_params(**params)
    #    return self


    #def get_params(self, deep=True):
    #    return self.transformer.get_params(deep)




class ColumnSelector(BaseEstimator, TransformerMixin):
    '''Select specified columns.

    Useful for freezing Feature Selection after subset search is ended.

    Parameters
    ----------
    columns : list of strings
        Columns to select.

    '''
    def __init__(self, columns=None):
        self.columns = columns


    def fit(self, X, y=None):
        '''Does nothing.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        self: ColumnSelector
            This estimator.

        '''
        if isinstance(self.columns, Iterable):
            self.columns = list(self.columns)
            # check if not all columns in X.columns
            diff = set(self.columns) - set(X.columns)
            assert not diff, 'Not all columns are in X: \n{}'.format(diff)
        else:
            self.columns = X.columns
        return self


    def transform(self, X):
        """Select specified columns from X.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        Xt : DataFrame, shape [n_samples, n_features]
            Transformed input.

        """
        try:
            return X[self.columns]

        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)




class TypeSelector(BaseEstimator, TransformerMixin):
    '''Select columns of specified type.

    Parameters
    ----------
    dtype : type

    '''
    def __init__(self, dtype):
        self.dtype = dtype


    def fit(self, X, y=None):
        '''Get names of columns of specified type.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        self

        '''
        if hasattr(self.dtype, '__iter__') and not isinstance(self.dtype, str):
            self.dtypes = self.dtype
        else:
            self.dtypes = [self.dtype]
        self.columns = X.select_dtypes(include=self.dtypes).columns
        return self


    def transform(self, X):
        """Select columns of specified type from X.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        Xt : DataFrame, shape [n_samples, n_features]
            Transformed input.

        """
        return X[self.columns]




class Identity(BaseEstimator, TransformerMixin):
    '''Dummy transformer.

    Just passes through its input.

    '''
    def __init__(self):
        pass


    def fit(self, X, y=None):
        '''Does nothing.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        self

        '''
        self.columns = X.columns
        return self


    def transform(self, X):
        """Pass X through.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X : DataFrame, shape [n_samples, n_features]
            Same data.

        """
        return X[self.columns]




class ColumnRenamer(BaseEstimator, TransformerMixin):
    '''Select columns of specified type.

    Parameters
    ----------
    columns : None or list of strings
        Columns to rename. If None, rename all.

    prefix, suffix : string
        String to concatenate at the beginning or/and at the end of original
        column name. Both equals to empty string ('') by default (do nothing).

    '''
    def __init__(self, columns=None, prefix='', suffix=''):
        self.columns = columns
        self.prefix = prefix
        self.suffix = suffix


    def fit(self, X, y=None):
        '''Creates mapper from old names to new names.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        self

        '''
        if not np.any(self.columns): self.columns = X.columns
        renamer = lambda column: '{}{}{}'.format(self.prefix, column, self.suffix)
        self.mapper = {column: renamer(column) for column in self.columns}
        return self


    def transform(self, X):
        """Renames selected columns.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        Xt : DataFrame, shape [n_samples, n_features]
            Same data.

        """
        return X.rename(self.mapper, axis='columns')




Imputer = lambda **params: PandasTransformer(impute.SimpleImputer(), **params)
