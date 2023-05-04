from typing import List, Iterable, Optional

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from category_encoders.utils import BaseEncoder
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import check_cv
from sklearn.utils.multiclass import type_of_target


def _smoothed_likelihood(
    x: pd.Series, y: pd.Series, smoothing: float, min_samples_leaf: int = 1
) -> pd.Series:
    """
    Computes the smoothed likelihood of the target variable y given the values of the feature x.

    Parameters
    ----------
    x : pd.Series)
        A pandas series representing the feature.
    y : pd.Series)
        A pandas series representing the target variable.
    smoothing : float
        A float representing the smoothing factor.
    min_samples_leaf : int, optional
        The minimum number of samples required to consider a split. Defaults to 1.

    Returns
    -------
    likelihood : pd.Series
        A pandas series containing the smoothed likelihoods.
    """
    # Compute the prior probability of y.
    prior = y.mean()

    # Compute the count and mean of y for each value of x.
    stats = y.groupby(x).agg(["count", "mean"])

    # Compute the smoothing factor for each value of x.
    smoove = 1 / (1 + np.exp(-(stats["count"] - min_samples_leaf) / smoothing))

    # Compute the smoothed likelihood for each value of x.
    likelihood = prior * (1 - smoove) + stats["mean"] * smoothing

    # If a value of x has only one sample, replace its smoothed likelihood with the prior probability.
    likelihood[stats["count"] == 1] = prior

    return likelihood


class TargetEncoder(TargetEncoder):
    """
    TargetEncoder is a class used to encode categorical features based on the target variable.

    Parameters:
    -----------
    verbose : int, default=0
        If > 0, prints the number of categorical columns encoded.
    cols : list of str or None, default=None
        List of column names to encode. If None, all categorical columns will be encoded.
    drop_invariant : bool, default=False
        If True, drops columns with zero variance.
    return_df : bool, default=True
        If True, returns a pandas DataFrame. Otherwise, a numpy array is returned.
    handle_missing : str, default='value'
        How to handle missing values. Options are 'value', 'return_nan', and 'raise'.
    handle_unknown : str, default='value'
         How to handle unknown categories. Options are 'value', 'return_nan', and 'raise'.
    min_samples_leaf : int, default=1
        The minimum number of samples required to perform smoothing.
    smoothing : float, default=1.0
            Smoothing effect to balance categorical average vs prior. Higher values of smoothing factor gives more
    weight to overall average.

    Attributes:
    -----------
    return_df : bool
        Whether to return a pandas DataFrame or not.
    drop_invariant : bool
        Whether to drop columns with zero variance or not.
    drop_cols : list of str
        List of column names dropped during fitting.
    verbose : int
        The level of verbosity.
    cols : list of str
        List of column names to encode.
    ordinal_encoder : OrdinalEncoder
        Ordinal encoder used to map categories to integers.
    min_samples_leaf : int
        The minimum number of samples required to perform smoothing.
    smoothing : float
        Smoothing effect to balance categorical average vs prior.
    _dim : int
        Number of columns in the input dataset.
    mapping : dict
        Mapping of the categorical variables and their respective target means.
    handle_unknown : str
        How to handle unknown categories.
    handle_missing : str
        How to handle missing values.
    _mean : float
        Global mean of the target variable.
    feature_names : list of str
        Names of the encoded features.
    """

    def __init__(
        self,
        verbose: int = 0,
        cols: Optional[List[str]] = None,
        drop_invariant: bool = False,
        return_df: bool = True,
        handle_missing: str = "value",
        handle_unknown: str = "value",
        min_samples_leaf: int = 1,
        smoothing: float = 1.0,
    ):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.ordinal_encoder = None
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing  # for python 3 only
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self._mean = None
        self.feature_names = None


class FastEncoder(BaseEstimator, TransformerMixin):
    """
    A transformer that applies smoothed likelihood encoding to categorical columns.

    Parameters
    ----------
    smoothing : float, default=1.0
        The smoothing factor to apply when calculating the encoding.
    min_samples_leaf : int, default=1
        The minimum number of samples required in each leaf for the tree-based method.

    Attributes
    ----------
    mapper : function
        A function that maps the encoded values to the original categorical values.
    """

    def __init__(self, smoothing: float = 1.0, min_samples_leaf: int = 1):
        self.mapper = None
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FastEncoder":
        """
        Fits the encoder to the training data.

        Parameters
        ----------
        X : pandas DataFrame
            The training data to fit the encoder to.
        y : pandas Series
            The target variable to encode against.

        Returns
        -------
        self : FastEncoder
            Returns the instance itself.
        """
        # Get the categorical columns from X
        cats = X.columns[X.dtypes.astype("str").isin(["object", "category"])]

        # Define a lambda function for the encoder
        encoder = lambda x: _smoothed_likelihood(
            x, y, self.min_samples_leaf, self.smoothing
        )

        # Create a dictionary of encoders for each categorical column
        encoders = {col: encoder(X[col].astype("str")) for col in cats}

        # Define a mapper function that maps the encoded values to the original categorical values
        self.mapper = (
            lambda x: x.astype("str").map(encoders[x.name]) if x.name in cats else x
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes the categorical columns in X.

        Parameters
        ----------
        X : pandas DataFrame
            The data to encode.

        Returns
        -------
        pandas DataFrame
            The encoded data.
        """

        # Apply the mapper function to each column in X
        return X.apply(self.mapper)


class EncoderCV(BaseEstimator):
    """
    A transformer that encodes features using a specified encoder and applies cross-validation.

    Parameters
    ----------
    encoder : object
        The encoder object that implements a 'fit' and 'transform' method.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation.
        - int, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used.
        In all other cases, KFold is used.
    n_jobs : int or None, optional
        Number of jobs to run in parallel. Default is None, which means 1 job.

    Attributes
    ----------
    encoders_ : list
        A list of encoder objects fitted on each fold.
    train_target : pandas.Series
        The target variable for the training set.
    train_shape_ : tuple
        The shape of the training data.
    train_index_ : pandas.Index
        The index of the training data.
    cv_ : cross-validation generator
        The cross-validation generator.
    """

    def __init__(self, encoder: object, cv: int = 5, n_jobs: Optional[int] = None):
        self.encoders_ = None
        self.encoder = encoder
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EncoderCV":
        """
        Fits the encoder on the training data using cross-validation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : pandas.Series
            The target variable.

        Returns
        -------
        self : EncoderCV
            Returns the encoder object.
        """

        # Fit the encoder on the training data using cross-validation
        self._fit_data(X, y)

        # Apply the encoder object to each fold of the data and obtain the fitted encoders
        self.encoders_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit)(clone(self.encoder), X, y, trn)
            for trn, oof in self._get_folds(X)
        )

        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fits the encoder on the training data using cross-validation and transforms the data.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit and transform.
        y : pandas.Series
            The target variable.

        Returns
        -------
        pandas.DataFrame
            The transformed data.
        """

        # Fit the encoder on the training data using cross-validation
        self._fit_data(X, y)

        # Apply the encoder object to each fold of the data and obtain the transformed data and fitted encoders
        paths = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_transform)(clone(self.encoder), X, y, trn, oof)
            for trn, oof in self._get_folds(X)
        )

        # Unpack the fitted encoders and transformed data, take the mean of the transformed data across folds, and return it
        self.encoders_, preds = zip(*paths)

        return self._mean_preds(preds)[X.columns]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data X by encoding categorical variables.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data to transform.

        Returns
        -------
        pandas.DataFrame
            Transformed data with encoded categorical variables.
        """

        # Check if the input data is training data and call the `transform_train` method
        if self._is_train(X):
            return self.transform_train(X)

        # Encode the input data using each fitted encoder in parallel
        preds = Parallel(n_jobs=self.n_jobs)(
            delayed(self._transform)(encoder, X) for encoder in self.encoders_
        )

        # Compute the mean of the encoded data
        return self._mean_preds(preds)[X.columns]

    def transform_train(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input training data X by encoding categorical variables.

        Parameters
        ----------
        X : pandas.DataFrame
            Input training data to transform.

        Returns
        -------
        pandas.DataFrame
            Transformed training data with encoded categorical variables.
        """

        # Encode the training data using each fitted encoder and its out-of-fold (oof) data in parallel
        preds = Parallel(n_jobs=self.n_jobs)(
            delayed(self._transform)(encoder, X, oof)
            for encoder, (_, oof) in zip(self.encoders_, self._get_folds(X))
        )

        # Compute the mean of the encoded
        return self._mean_preds(preds)[X.columns]

    def _mean_preds(self, preds: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Compute the mean of the predictions across all folds.

        Parameters
        ----------
        preds : list of pandas.DataFrame
            List of predictions, where each element of the list is a pandas DataFrame
            with the same columns as the original data X.

        Returns
        -------
        pandas.DataFrame
            The mean of the predictions across all folds.
        """
        return pd.concat(preds, axis=1).groupby(level=0, axis=1).mean()

    def _fit(
        self, encoder: BaseEncoder, X: pd.DataFrame, y: pd.Series, trn: np.ndarray
    ) -> object:
        """
        Fit the encoder on the training data.

        Parameters
        ----------
        encoder : estimator
            The encoder to fit.
        X : pandas.DataFrame
            The input data X.
        y : pandas.Series
            The target variable y.
        trn : numpy.ndarray
            The indices of the training data.

        Returns
        -------
        estimator
            The fitted encoder.
        """
        return encoder.fit(X.iloc[trn], y.iloc[trn])

    def _fit_transform(
        self,
        encoder: BaseEncoder,
        X: pd.DataFrame,
        y: pd.Series,
        trn: np.ndarray,
        oof: np.ndarray,
    ) -> tuple[object, pd.DataFrame]:
        """
        Fit and transform the encoder on the training and validation data.

        Parameters
        ----------
        encoder : estimator
            The encoder to fit and transform.
        X : pandas.DataFrame
            The input data X.
        y : pandas.Series
            The target variable y.
        trn : numpy.ndarray
            The indices of the training data.
        oof : numpy.ndarray
            The indices of the validation data.

        Returns
        -------
        Tuple[estimator, pandas.DataFrame]
            The fitted encoder and the transformed validation data.
        """
        Xt = encoder.fit(X.iloc[trn], y.iloc[trn]).transform(X.iloc[oof])
        return encoder, Xt

    def _transform(self, encoder: BaseEncoder, X: pd.DataFrame) -> np.ndarray:
        """
        Applies the provided encoder to the given data and returns the encoded result.

        Parameters
        ----------
        encoder : BaseEncoder
            An encoder object with a `transform` method to encode the data.
        X : pd.DataFrame
            The data to encode.

        Returns
        -------
        encoder : np.ndarray
            Encoded data as numpy.ndarray.
        """
        return encoder.transform(X)

    def _transform_train(
        self, encoder: BaseEncoder, X: pd.DataFrame, oof: np.ndarray
    ) -> np.ndarray:
        """
        Applies the provided encoder to the training data based on the out-of-fold index and returns the encoded result.

        Parameters
        ----------
        encoder : object
            An encoder object with a `transform` method to encode the data.
        X : DataFrame
            The data to encode.
        oof : ndarray
            An array of indices for the out-of-fold data.

        Returns
        -------
        encoder : ndarray
            Encoded data as numpy.ndarray.
        """
        return encoder.transform(X.iloc[oof])

    def _fit_data(self, X: pd.DataFrame, y: pd.Series) -> "EncoderCV":
        """
        Fits the training data for the model.

        Parameters
        ----------
        X : DataFrame
            The training data.
        y : Series
            The target values for the training data.

        Returns
        -------
        self : EncoderCV
            The fitted model object.
        """
        # Fit train dataset
        self.train_target = y.copy()
        self.train_shape_ = X.shape
        self.train_index_ = X.index

        # Define cross-validation
        task_type = type_of_target(self.train_target)

        if task_type == "binary":
            classifier = True
        elif task_type == "continuous":
            classifier = False
        else:
            raise ValueError("Unsupported task type '{}'".format(task_type))

        self.cv_ = check_cv(self.cv, self.train_target, classifier)

        return self

    def _get_folds(self, X: pd.DataFrame) -> Iterable:
        """
        Splits the data into folds for cross-validation.

        Parameters
        ----------
        X : DataFrame
            The data to split into folds.

        Returns
        -------
        iterable : object
            An iterable object with the train and validation indices for each fold.
        """
        return self.cv_.split(X, self.train_target)

    def _is_train(self, X: pd.DataFrame) -> bool:
        """
        Checks if the given data is the same as the training data.

        Parameters
        ----------
        X : DataFrame
            The data to check.

        Returns
        -------
        bool :
            A boolean indicating if the data is the same as the training data.
        """
        return (X.shape is self.train_shape_) and (X.index is self.train_index_)


class NaiveBayesEncoder(BaseEstimator, TransformerMixin):
    """
    Naive Bayes Encoder transformer.
    Transforms a sparse matrix of categorical features into a matrix of
    log-odds ratios of class probabilities for each feature.

    Parameters
    ----------
    smooth : float, default=5.0
        Laplace smoothing parameter to avoid zero probabilities.

    Attributes
    ----------
    _r : numpy.ndarray, shape (n_features,)
        Log-odds ratio of class probabilities for each feature.
    """

    def __init__(self, smooth: float = 5.0):
        self._r = None
        self.smooth = smooth

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the input sparse matrix X into a matrix of log-odds ratios
        of class probabilities for each feature using the learned _r values.

        Parameters
        ----------
        X : sparse matrix, shape (n_samples, n_features)
            Input sparse matrix of categorical features.

        Returns
        -------
        X_log_odds : sparse matrix, shape (n_samples, n_features)
            Transformed sparse matrix of log-odds ratios of class probabilities
            for each feature.
        """
        return X.multiply(self._r)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveBayesEncoder":
        """
        Fit the transformer to the training data X and labels y.
        Calculates the log-odds ratio _r of class probabilities for each feature.

        Parameters
        ----------
        X : pd.DataFrame
            Input sparse matrix of categorical features.

        y : pd.Series
            Target variable.

        Returns
        -------
        self : object
            Returns self.
        """
        self._r = np.log(self._pr(X, y, 1) / self._pr(X, y, 0))
        return self

    def _pr(self, X: pd.DataFrame, y: pd.Series, val: int) -> np.ndarray:
        """
        Private method that calculates the probability distribution of
        each feature for class y=val using Laplace smoothing.

        Parameters
        ----------
        X : sparse matrix, shape (n_samples, n_features)
            Input sparse matrix of categorical features.

        y : array-like, shape (n_samples,)
            Target variable.

        val : int
            Class value (0 or 1).

        Returns
        -------
        prob : numpy.ndarray, shape (n_features,)
            Probability distribution of each feature for class y=val.
        """
        prob = X[y == val].sum(0)
        return (prob + self.smooth) / ((y == val).sum() + self.smooth)
