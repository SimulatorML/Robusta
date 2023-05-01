from collections import defaultdict
from itertools import chain
from typing import Optional, Union, Tuple, Generator

import numpy as np
import pandas as pd
from sklearn.base import clone, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GroupKFold, KFold
from sklearn.utils import check_random_state


def shuffle_labels(labels: pd.Series,
                   random_state: Optional[int] = None) -> pd.Series:
    """
    Shuffle the labels of a pandas Series with unique values.

    Parameters
    ----------
    labels : pd.Series
        The list containing the labels to be shuffled.
    random_state : int, optional
        Seed or state for the random number generator. Default is None, which uses the global numpy
        random number generator.

    Returns
    -------
    labels : pd.Series
        A new pandas Series object with the same values as the original labels, but with the values shuffled
        using the random state specified.

    Raises
    ------
    TypeError:
        If the input `labels` is not a pandas Series.
    ValueError:
        If the input `labels` contains non-unique values.
    """

    # Create a random number generator with the given random_state
    rstate = np.random.RandomState(seed=random_state)

    # Get the unique labels in the input Series
    unique_labels = np.unique(labels)

    # Shuffle the unique labels using the random number generator
    random_labels = rstate.permutation(unique_labels)

    # Create a dictionary mapping original labels to shuffled labels
    mapper = dict(zip(unique_labels, random_labels))

    # Map the original labels to the shuffled labels and return the result as a new Series
    return labels.map(mapper)


class RepeatedGroupKFold:
    """
    Repeated GroupKFold cross-validation with shuffled group labels.

    This class implements repeated cross-validation with shuffled group
    labels, using the GroupKFold strategy to split the data into groups.
    The splitting process is repeated `n_repeats` times, shuffling the group
    labels each time to obtain different splits. For each split, the training
    data is defined as the union of all the folds except the test fold, and the
    test data is the fold that is left out.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=3
        Number of times to repeat the splitting process with shuffled group
        labels.
    random_state : int, default=0
        Seed used by the random number generator.

    Examples
    --------
    >>> rgkf = RepeatedGroupKFold(n_splits=3, n_repeats=2, random_state=42)
    >>> X = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ...                   'B': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd']})
    >>> y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    >>> groups = pd.Series(['g1', 'g1', 'g1', 'g1', 'g2', 'g2', 'g2', 'g3', 'g3', 'g3'])
    >>> for train_idx, test_idx in rgkf.split(X, y, groups):
    ...     print(f'Train: {train_idx}, Test: {test_idx}')
    Train: [0 1 2 4 5 6 7 8], Test: [3]
    Train: [0 1 2 3 4 5 8 9], Test: [6 7]
    Train: [3 6 7 8 9 0 1 2], Test: [4 5]
    Train: [0 1 2 3 4 5 6 7], Test: [8 9]
    Train: [4 5 6 7 8 9 0 1], Test: [2 3]
    Train: [2 3 4 5 6 7 8 9], Test: [0 1]
    """

    def __init__(self,
                 n_splits: int = 5,
                 n_repeats: int = 3,
                 random_state: int = 0):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self,
              X: pd.DataFrame,
              y: pd.Series,
              groups: pd.Series) -> chain[Tuple[np.array, np.array]]:
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The data to be split into train and test sets.
        y : pandas Series of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : pandas Series of shape (n_samples,)
            The groups to use for group-wise cross-validation. Each sample
            belongs to a group, and the groups should not overlap across the
            folds.

        Yields
        ------
        train_index : numpy array
            The indices of the training set.
        test_index : numpy array
            The indices of the test set.
        """

        # Create an empty list to store the splits
        splits = []

        # Repeat the splitting process for n_repeats
        for i in range(self.n_repeats):
            # Shuffle the group labels for each repeat
            groups_i = shuffle_labels(labels=groups,
                                      random_state=self.random_state + i)

            # Perform GroupKFold cross-validation with n_splits
            cv = GroupKFold(self.n_splits)

            # Obtain the train and test indices for each split
            split = cv.split(X=X,
                             y=y,
                             groups=groups_i)

            # Append the split indices to the splits list
            splits.append(split)

        # Return the chain of split indices
        return chain(*splits)

    def get_n_splits(self) -> int:
        """
        Return the number of splitting iterations in the cross-validator.

        Returns
        -------
        n_splits : int
            The number of splitting iterations in the cross-validator.
        """
        return self.n_repeats * self.n_splits


class RepeatedKFold:
    """
    Repeated K-Fold cross-validator.

    Initialize the RepeatedKFold cross-validator.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=3
        Number of times the cross-validator repeats the K-Fold splitting
        procedure.
    random_state : int or None, default=None
        Seed used by the random number generator.
    """

    def __init__(self,
                 n_splits: int = 5,
                 n_repeats: int = 3,
                 random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self,
              X: pd.DataFrame,
              y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate indices to split data into training and validation set.

        Parameters
        ----------
        X : pandas.DataFrame
            Data to be split.
        y : pandas.Series
            Target variable to be used to split the data.

        Yields
        ------
        train_index : np.ndarray
            The indices of the training data for each split.
        valid_index : np.ndarray
            The indices of the validation data for each split.
        """

        # Create a random state object using the specified random state or a new one
        rstate = check_random_state(seed=self.random_state)

        # Loop over the number of repeats
        for _ in range(self.n_repeats):
            # Generate a new random seed for each repeat
            seed = rstate.randint(2 ** 32 - 1)

            # Create a KFold object with the specified number of splits, shuffled each time with the new seed
            cv = KFold(n_splits=self.n_splits,
                       shuffle=True,
                       random_state=seed)

            # Loop over the training and validation indices for each split
            for trn, oof in cv.split(X, y):
                # Yield (i.e., return) the training and validation indices for the current split
                yield trn, oof

    def get_n_splits(self) -> int:
        """
        Return the number of splits.

        Returns
        -------
        n_splits : int
            The total number of splits (i.e., repeats times splits).
        """

        # Return the total number of splits (i.e., repeats times splits)
        return self.n_repeats * self.n_splits


class StratifiedGroupKFold:
    """
    StratifiedGroupKFold is a variation of K-fold cross-validation that ensures both stratification by label and
    grouping by sample. This is particularly useful for machine learning tasks where there are multiple samples
    from each group, and the samples within each group may be similar to each other. The algorithm works by dividing
    the data into batches based on the number of unique groups, and then sorting the batches by the standard deviation
    of the counts of each label within each batch. The algorithm then assigns each batch to one of the K folds,
    ensuring that each fold contains a proportional number of samples from each label, as well as a proportional number
    of samples from each group.

    Parameters
    ----------
    n_splits : int, default=5
        The number of folds to generate.
    n_batches : int, default=1024
        The number of batches to generate.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting.
    random_state : int, default=0
        Random seed to use for shuffling.

    Attributes
    ----------
    n_splits : int
        The number of splits.
    n_batches : int
        The number of batches.
    shuffle : bool
        Whether the data is shuffled.
    random_state : int
        Random seed for shuffling.
    """

    def __init__(self,
                 n_splits: int = 5,
                 n_batches: int = 1024,
                 shuffle: bool = False,
                 random_state: int = 0):
        self.n_splits = n_splits
        self.n_batches = n_batches
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self) -> int:
        """
        Returns the number of splits.

        Returns
        -------
        int
            The number of splits.

        """
        return self.n_splits

    def split(self,
              y: pd.Series,
              groups: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Generate indices to split data into training and validation sets.

        Parameters
        ----------
        y : pd.Series
            The target variable to be used for stratification.
        groups : pd.Series
            The array containing the groups for each sample.

        Yields
        ------
        Tuple[pd.Series, pd.Series]
            The training and validation indices for each split.

        """

        # Unique groups
        groups_unique = set(groups.unique())

        # Unique labels
        labels = np.sort(y.unique())

        # Count of each group for each label
        counts = [groups[y == label].value_counts(sort=False) for label in labels]

        # Concatenate counts and fill missing values with 0
        counts = pd.concat(counts, axis=1).fillna(0).astype(int)

        # Rename columns to labels
        counts.columns = labels

        if self.shuffle:
            # Shuffle the counts if specified
            counts = counts.sample(frac=1, random_state=self.random_state)

        # Number of unique groups
        n = len(groups_unique)

        # Number of groups in each batch
        batch_size = max(n // self.n_batches, 1)

        # Split counts into batches
        batches = [counts.iloc[k:k + batch_size] for k in range(0, n, batch_size)]

        # Sort batches by decreasing standard deviation of counts
        batches.sort(key=lambda batch: -batch.sum().std())

        # Initialize labels for each fold
        fold_labels = pd.DataFrame(0, columns=labels, index=range(self.n_splits))

        # Initialize groups for each fold as a set
        fold_groups = defaultdict(set)

        # Loop through each batch
        for batch in batches:
            # Groups in the current batch
            batch_groups = batch.index

            # Total counts for each label in the current batch
            batch_labels = batch.sum()

            # Index of the best fold
            best_idx = None

            # Standard deviation of labels for the best fold
            best_std = None

            # Loop through each fold
            for i in range(self.n_splits):
                # Add the batch labels to the current fold
                fold_labels.loc[i] += batch_labels

                # Calculate the mean standard deviation of labels for all folds
                fold_std = fold_labels.std().mean()

                # Update the best fold if a better one is found
                if best_std is None or fold_std < best_std:
                    best_std = fold_std
                    best_idx = i
                # Subtract the batch labels from the current fold
                fold_labels.loc[i] -= batch_labels
            # Add the batch labels to the best fold
            fold_labels.loc[best_idx] += batch_labels

            # Add the batch groups to the best fold
            fold_groups[best_idx].update(batch_groups)

        # Loop through the out-of-fold (oof) groups for each fold
        for oof_groups in fold_groups.values():
            # Groups in the training set
            trn_groups = groups_unique - oof_groups

            # Index of the training set
            trn = groups[groups.isin(trn_groups)].index

            # Get the OOF indices
            oof = groups[groups.isin(oof_groups)].index

            # Yield the training and OOF indices
            yield trn, oof


class RepeatedStratifiedGroupKFold:
    """
    Repeated Stratified Group K-Fold cross-validator.

    Provides train/test indices to split data in train/test sets. This cross-validation object is a variation of
    StratifiedKFold, which returns stratified folds while preserving the percentage of samples in each group.
    Additionally, it repeats the cross-validation process n_repeats times and uses batches of groups to create
    folds with similar standard deviation of class distribution.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=3
        Number of times cross-validator needs to be repeated.
    n_batches : int, default=1024
        Number of batches to divide the groups into.
    random_state : int or None, default=None
        Seed used by the random number generator.

    Attributes
    ----------
    n_splits : int
        The number of splits used in cross-validation.
    n_repeats : int
        The number of times cross-validator needs to be repeated.
    n_batches : int
        The number of batches to divide the groups into.
    random_state : int or None
        The seed used by the random number generator.
    """

    def __init__(self,
                 n_splits: int = 5,
                 n_repeats: int = 3,
                 n_batches: int = 1024,
                 random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_batches = n_batches
        self.random_state = random_state

    def get_n_splits(self) -> int:
        """
        Returns the number of splits.

        Returns
        -------
        int
            The number of splits.

        """
        return self.n_splits

    def split(self,
              y: pd.Series,
              groups: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Generate indices to split data into training and validation set.

        Parameters
        ----------
        y : pd.Series, shape (n_samples,)
            The target variable for supervised learning problems.
        groups : pd.Series, shape (n_samples,)
            Group labels for the samples.

        Returns
        -------
        generator
            Yields the training and validation indices for each fold.
        """
        # Get unique groups
        groups_unique = set(groups.unique())

        # Get unique labels and sort them
        labels = np.sort(y.unique())

        # Count number of samples in each group for each label
        counts = [groups[y == label].value_counts(sort=False) for label in labels]

        # Concatenate counts into a dataframe, fill NAs with 0, and cast to integer
        counts = pd.concat(counts, axis=1).fillna(0).astype(int)

        # Set the columns of the counts dataframe to the labels
        counts.columns = labels

        # Repeat the process for n_repeats times
        for _ in range(self.n_repeats):
            # Shuffle the counts dataframe
            counts = counts.sample(frac=1, random_state=self.random_state)

            # Calculate the number of groups and the batch size
            n = len(groups_unique)
            batch_size = max(n // self.n_batches, 1)

            # Divide the counts dataframe into batches of similar size
            batches = [counts.iloc[k:k + batch_size] for k in range(0, n, batch_size)]

            # Sort the batches in descending order of their standard deviation
            batches.sort(key=lambda batch: -batch.sum().std())

            # Initialize dataframes to store the fold labels and groups
            fold_labels = pd.DataFrame(0, columns=labels, index=range(self.n_splits))
            fold_groups = defaultdict(set)

            # Iterate through each batch
            for batch in batches:
                # Get the group indices of the current batch
                batch_groups = batch.index

                # Get the label counts of the current batch
                batch_labels = batch.sum()

                # Initialize variables to store the best fold index and standard deviation
                best_idx = None
                best_std = None

                # Iterate through each fold
                for i in range(self.n_splits):
                    # Add the label counts of the current batch to the current fold
                    fold_labels.loc[i] += batch_labels

                    # Calculate the standard deviation of the label counts across all folds
                    fold_std = fold_labels.std().mean()

                    # Update the best fold index and standard deviation if the current fold has a lower
                    # standard deviation
                    if best_std is None or fold_std < best_std:
                        best_std = fold_std
                        best_idx = i

                    # Remove the label counts of the current batch from the current fold
                    fold_labels.loc[i] -= batch_labels

                # Add the label counts of the current batch to the best fold
                fold_labels.loc[best_idx] += batch_labels

                # Add the group indices of the current batch to the best fold
                fold_groups[best_idx].update(batch_groups)

            # Iterate through each out-of-fold group
            for oof_groups in fold_groups.values():
                # Get the indices of the training groups
                trn_groups = groups_unique - oof_groups

                # Get the indices of the training samples
                trn = groups[groups.isin(trn_groups)].index

                # Get the indices of the validation samples
                oof = groups[groups.isin(oof_groups)].index

                # Yield the training and validation indices as a tuple
                yield trn, oof


class AdversarialValidation:
    """
    Adversarial validation is a technique used to assess whether two datasets come from the same distribution. In the
    context of machine learning, this can be useful to detect when the training and test datasets are too different,
    which can lead to poor model performance when it's deployed in production. This class provides an implementation
    of adversarial validation using a pre-fitted classifier to predict whether each sample in the input dataframe is
    more likely to come from the training set or the validation set.

    Parameters
    ----------
    classifier: ClassifierMixin
        A pre-fitted classifier that will be used to predict whether each sample in the input dataframe is more
        likely to come from the training set or the validation set.
    train_size: Optional[float, int] = None
        The proportion or absolute number of samples to use for the training set. If both train_size and test_size
        are None, train_size defaults to 80% of the input dataframe size.
    test_size: Optional[float, int] = None
        The proportion or absolute number of samples to use for the validation set. If both train_size and test_size
        are None, train_size defaults to 20% of the input dataframe size.

    Attributes
    ----------
    classifier: ClassifierMixin
        The pre-fitted classifier used to predict whether each sample in the input dataframe is more likely to come
        from the training set or the validation set.
    train_size: Optional[float, int]
        The proportion or absolute number of samples to use for the training set.
    test_size: Optional[float, int]
        The proportion or absolute number of samples to use for the validation set.
    """

    def __init__(self,
                 classifier: ClassifierMixin,
                 train_size: Optional[Union[float, int]] = None,
                 test_size: Optional[Union[float, int]] = None):
        self.classifier = classifier
        self.train_size = train_size
        self.test_size = test_size

    def split(self,
              X: pd.DataFrame) -> Generator:
        """
        Split the input dataframe into training and validation sets using adversarial validation.
        Returns a generator of tuples containing the indices of the training and validation sets.

        Parameters
        ----------
        X: pd.DataFrame
            The input dataframe to split into training and validation sets.

        Raises
        ------
        NotFittedError:
            If the passed classifier has not been pre-fitted.

        Returns
        -------
        generator:
            A generator of tuples containing the indices of the training and validation sets.
        """

        # checks if the classifier has been fitted yet
        if not hasattr(self.classifier, 'classes_'):
            raise NotFittedError('Passed classifier must be pre-fitted')

        # gets the train size based on the passed train_size or test_size and the input dataframe size
        train_size = self._get_train_size(X)

        # gets the predicted probabilities of the classifier on the input dataframe
        proba = self.classifier.predict_proba(X)

        # sorts the predicted probabilities of the positive class (1) in ascending order and returns
        # the indices of the sorted elements
        ranks = proba[:, 1].argsort()

        # yields a tuple of train indices and validation indices based on the train_size and input dataframe size
        yield ranks[:train_size], ranks[train_size:]

    @staticmethod
    def get_n_splits():
        """
        Returns the number of splits.
        """
        return 1

    def _get_train_size(self,
                        X: pd.DataFrame) -> int:
        """
        Helper function to get the size of the training set based on the passed train_size and test_size parameters.

        Parameters
        ----------
        X : pd.DataFrame
            The input dataframe to split.

        Returns
        --------
        training_size : int
            The size of the training set.
        """

        # gets the length of the input dataframe
        size = len(X)

        # gets the train_size attribute of the object
        train_size = self.train_size

        # gets the test_size attribute of the object
        test_size = self.test_size
        if train_size is not None and test_size is not None:
            raise ValueError("train_size and test_size could not be set both")

        # if neither train_size nor test_size is passed, sets train_size to 80% of the input dataframe size
        if train_size is None and test_size is None:
            return size - int(size * 0.2)

        if train_size is not None:
            if isinstance(train_size, float):
                if 0 < train_size < 1:
                    # if train_size is a float between 0 and 1, returns the integer train_size based on
                    # the input dataframe size
                    return int(size * train_size)
                else:
                    raise ValueError("Float train_size must be in range (0, 1). "
                                     "Passed {}".format(train_size))

            elif isinstance(train_size, int):
                if 0 < train_size < size:
                    # if train_size is an integer between 1 and the input dataframe size, returns the train_size
                    return train_size
                else:
                    raise ValueError("Integer train_size must be in range [1, {}]. "
                                     "Passed {}".format(size, train_size))
            else:
                raise ValueError("Unknown type of train_size passed {}".format(train_size))

        # If test_size is set, handle different types of input values for test_size
        if test_size is not None:
            if isinstance(test_size, float):
                # If test_size is a float, make sure it is in the range (0, 1), and set
                # train_size to be (1 - test_size) proportion of X
                if 0 < test_size < 1:
                    return size - int(size * test_size)
                else:
                    # Raise ValueError if test_size is neither a float nor an integer
                    raise ValueError("Float test_size must be in range (0, 1). "
                                     "Passed {}".format(test_size))
            elif isinstance(test_size, int):
                if 0 < test_size < size:
                    return size - test_size
                else:
                    raise ValueError("Integer test_size must be in range [1, {}]. "
                                     "Passed {}".format(size, test_size))
            else:
                raise ValueError("Unknown type of test_size passed {}".format(test_size))

def make_adversarial_validation(classifier: ClassifierMixin,
                                X_train: pd.DataFrame,
                                X_test: pd.DataFrame,
                                train_size: Optional[Union[float, int]] = None,
                                test_size: Optional[Union[float, int]] = None) -> AdversarialValidation:
    """
    Returns an AdversarialValidation object that can be used to split data into train and validation sets based on a
    pre-fitted binary classifier's prediction probabilities. The goal of this split is to ensure that the train and
    validation sets have similar feature distributions, which can be useful in detecting and mitigating dataset shift
    between training and validation data.

    Parameters
    ----------
    classifier: ClassifierMixin
        A pre-fitted binary classifier (e.g. a logistic regression or a random forest classifier).
    X_train: pd.DataFrame
        The training data as a pandas DataFrame.
    X_test: pd.DataFrame
        The validation data as a pandas DataFrame.
    train_size: Optional[Union[float, int]]
        The proportion of the combined dataset to be used as the training set. This can be a float (between
        0 and 1) to indicate the proportion of the combined dataset to use as training data, or an integer to
        indicate the exact number of samples to use for training. If both train_size and test_size are None (the
        default), train_size is set to 80% of the combined dataset size.
    test_size: Optional[Union[float, int]]
        The proportion of the combined dataset to be used as the validation set. This can be a float (between
        0 and 1) to indicate the proportion of the combined dataset to use as validation data, or an integer to
        indicate the exact number of samples to use for validation. If both train_size and test_size are None (the
        default), train_size is set to 80% of the combined dataset size.

    Returns
    -------
    class : AdversarialValidation
        An AdversarialValidation object that can be used to generate train and validation sets based on the
        pre-fitted binary classifier's prediction probabilities.

    Raises
    ------
    ValueError:
        If both train_size and test_size are passed.
    """
    # Concatenate the training and test data into a single DataFrame
    X = pd.concat([X_train, X_test])

    # Create binary labels for the training and test data (0 for training, 1 for test)
    y = [0] * len(X_train) + [1] * len(X_test)

    # Fit the classifier on the combined dataset with binary labels
    clf = clone(classifier).fit(X, y)

    # Create and return an AdversarialValidation object with the trained classifier and specified train and test sizes
    return AdversarialValidation(classifier=clf, train_size=train_size, test_size=test_size)