import numpy as np
import pandas as pd

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import NotFittedError

from collections import Counter, defaultdict
from itertools import chain

__all__ = [
    'StratifiedGroupKFold',
    'RepeatedGroupKFold',
    'RepeatedStratifiedGroupKFold',
    'AdversarialValidation',
    'make_adversarial_validation',
]



def shuffle_labels(labels, random_state=None):
    rstate = np.random.RandomState(random_state)
    unique_labels = np.unique(labels)
    random_labels = rstate.permutation(unique_labels)
    mapper = dict(zip(unique_labels, random_labels))
    return labels.map(mapper)



class RepeatedGroupKFold():
    """Repeated Group KFold

    Same as RepeatedKFold but each group presents only in one fold on each repeat.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.

    n_repeats : int, optional
        Number of times cross-validator needs to be repeated.

    random_state : int, RandomState instance or None, optional, default=None
        If None, the random number generator is the RandomState instance used by np.random.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;

    """
    def __init__(self, n_splits=5, n_repeats=3, random_state=0):

        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y, groups):

        splits = []

        for i in range(self.n_repeats):

            groups_i = shuffle_labels(groups, self.random_state + i)
            cv = GroupKFold(self.n_splits)

            split = cv.split(X, y, groups_i)
            splits.append(split)

        return chain(*splits)

    def get_n_splits(self):
        return self.n_repeats * self.n_splits



class StratifiedGroupKFold():
    """Stratified Group KFold

    Same as StratifiedKFold but each group presents only in one fold.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.

    n_batches : int, default=1024
        Split groups to min(n_batches, n_groups) parts.
        Must be greater than n_splits.

    shuffle : boolean, optional
        Whether to shuffle each class’s samples before splitting into batches.

    random_state : int, RandomState instance or None, optional, default=0
        If None, the random number generator is the RandomState instance used by np.random.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;

    """
    def __init__(self, n_splits=5, n_batches=1024, shuffle=False, random_state=0):
        self.n_splits = n_splits
        self.n_batches = n_batches
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y, groups):

        # Global stats
        groups_unique = set(groups.unique())
        labels = np.sort(y.unique())

        counts = [groups[y == label].value_counts(sort=False) for label in labels]
        counts = pd.concat(counts, axis=1).fillna(0).astype(int)
        counts.columns = labels

        labels_total = counts.sum()

        if self.shuffle:
            counts = counts.sample(frac=1, random_state=self.random_state)

        # Mini-Batches
        n = len(groups_unique)
        batch_size = max(n // self.n_batches, 1)

        batches = [counts.iloc[k:k+batch_size] for k in range(0, n, batch_size)]
        batches.sort(key=lambda batch: -batch.sum().std())

        # Local stats (per fold)
        fold_labels = pd.DataFrame(0, columns=labels, index=range(self.n_splits))
        fold_groups = defaultdict(set)

        for batch in batches:
            batch_groups = batch.index
            batch_labels = batch.sum()

            best_idx = None
            best_std = None

            for i in range(self.n_splits):

                fold_labels.loc[i] += batch_labels
                fold_std = fold_labels.std().mean()

                if best_std is None or fold_std < best_std:
                    best_std = fold_std
                    best_idx = i

                fold_labels.loc[i] -= batch_labels

            fold_labels.loc[best_idx] += batch_labels
            fold_groups[best_idx].update(batch_groups)

        # Yield indices
        for oof_groups in fold_groups.values():
            trn_groups = groups_unique - oof_groups

            trn = groups[groups.isin(trn_groups)].index
            oof = groups[groups.isin(oof_groups)].index

            yield trn, oof


class RepeatedStratifiedGroupKFold():
    """Repeated Stratified Group KFold

    Same as RepeatedStratifiedKFold but each group presents only in one fold on each repeat.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.

    n_repeats : int, optional
        Number of times cross-validator needs to be repeated.

    n_batches : int, default=1024
        Split groups to min(n_batches, n_groups) parts.
        Must be greater than n_splits.

    random_state : int, RandomState instance or None, optional, default=0
        If None, the random number generator is the RandomState instance used by np.random.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;

    """
    def __init__(self, n_splits=5, n_repeats=3, n_batches=1024, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_batches = n_batches
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats

    def split(self, X, y, groups):

        # Global stats
        groups_unique = set(groups.unique())
        labels = np.sort(y.unique())

        counts = [groups[y == label].value_counts(sort=False) for label in labels]
        counts = pd.concat(counts, axis=1).fillna(0).astype(int)
        counts.columns = labels

        labels_total = counts.sum()

        for _ in range(self.n_repeats):
            counts = counts.sample(frac=1, random_state=self.random_state)

            # Mini-Batches
            n = len(groups_unique)
            batch_size = max(n // self.n_batches, 1)

            batches = [counts.iloc[k:k+batch_size] for k in range(0, n, batch_size)]
            batches.sort(key=lambda batch: -batch.sum().std())

            # Local stats (per fold)
            fold_labels = pd.DataFrame(0, columns=labels, index=range(self.n_splits))
            fold_groups = defaultdict(set)

            for batch in batches:
                batch_groups = batch.index
                batch_labels = batch.sum()

                best_idx = None
                best_std = None

                for i in range(self.n_splits):

                    fold_labels.loc[i] += batch_labels
                    fold_std = fold_labels.std().mean()

                    if best_std is None or fold_std < best_std:
                        best_std = fold_std
                        best_idx = i

                    fold_labels.loc[i] -= batch_labels

                fold_labels.loc[best_idx] += batch_labels
                fold_groups[best_idx].update(batch_groups)

            # Yield indices
            for oof_groups in fold_groups.values():
                trn_groups = groups_unique - oof_groups

                trn = groups[groups.isin(trn_groups)].index
                oof = groups[groups.isin(oof_groups)].index

                yield trn, oof



class AdversarialValidation():
    """Adversarial Validation

    Holdout split by the train/test similarity. Inner ``classifier`` must be
    already fitted to the concatenated dataset with binary target, where 1 means
    test set and 0 – train set. Provides list with single train/oof indices,
    where oof – subset of size ``test_size`` with maximum class 1 probability.

    Parameters
    ----------
    classifier : estimator object
        Fitted estimator for train/test similarity measurement.
        Class 1 for test set and 0 for train.

    train_size : float, int, or None (default=None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    test_size : float, int, None (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.2.

    """

    def __init__(self, classifier, train_size=None, test_size=None):

        self.classifier = classifier
        self.train_size = train_size
        self.test_size = test_size


    def split(self, X, y, groups=None):

        if not hasattr(self.classifier, 'classes_'):
            raise NotFittedError('Passed classifier must be pre-fitted')

        train_size = self._get_train_size(X)

        proba = self.classifier.predict_proba(X)
        ranks = proba[:, 1].argsort()

        yield ranks[:train_size], ranks[train_size:]


    def get_n_splits(self):
        return 1


    def _get_train_size(self, X):

        size = len(X)

        train_size = self.train_size
        test_size = self.test_size

        if train_size is not None and test_size is not None:
            raise ValueError("train_size and test_size could not be set both")

        if train_size is None and test_size is None:
            return size - int(size * 0.2)

        if train_size is not None:
            if isinstance(train_size, float):
                if 0 < train_size < 1:
                    return int(size * train_size)
                else:
                    raise ValueError("Float train_size must be in range (0, 1). "
                                     "Passed {}".format(train_size))

            elif isinstance(train_size, int):
                if 0 < train_size < size:
                    return train_size
                else:
                    raise ValueError("Integer train_size must be in range [1, {}]. "
                                     "Passed {}".format(size, train_size))

            else:
                raise ValueError("Unknown type of train_size passed {}".format(train_size))

        if test_size is not None:
            if isinstance(test_size, float):
                if 0 < test_size < 1:
                    return size - int(size * test_size)
                else:
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



def make_adversarial_validation(classifier, X_train, X_test, train_size=None, test_size=None):
    """Construct AdversarialValidation object from unfitted classifier.

    See AdversarialValidation documentation for details.

    Parameters
    ----------
    classifier : estimator object
        Estimator for train/test similarity measurement.
        Would be fitted on concatenated X_train/X_test dataset.

    train_size : float, int, or None (default=None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    test_size : float, int, None (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.2.

    """

    X = pd.concat([X_train, X_test])
    y = [0]*len(X_train) + [1]*len(X_test)

    return AdversarialValidation(clone(classifier).fit(X, y),
                                 train_size=train_size,
                                 test_size=test_size)
