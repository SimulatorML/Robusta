from copy import copy
from typing import Union, Optional, Sequence, Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.random import check_random_state


class FeatureSubset:
    """
    A class to represent a subset of features.

    Args:
        features: A numpy array or pandas Index containing all features.
        subset: A list or tuple of indices or feature names to be included in the subset. Defaults to None.
        mask: A boolean list or numpy array indicating which features are selected. Defaults to None.
        group: A boolean flag to indicate if the features represent groups (e.g. multi-level columns). Defaults to False.

    Raises:
        ValueError: If both <subset> and <mask> are set.
        AssertionError: If features are not unique.
    """
    def __init__(self,
                 features: Union[pd.Index, np.ndarray],
                 subset: Optional[Sequence[Union[int, str]]] = None,
                 mask: Optional[Sequence[bool]] = None,
                 group: bool = False):

        # Features
        msg = '<features> must be unique'
        assert len(set(features)) == len(features), msg

        if group:
            self.features = features.get_level_values(0).unique()
            self.group = True
        else:
            self.features = np.array(features)
            self.group = False

        # subset OR mask
        if subset is not None and mask is not None:
            raise ValueError('<subset> & <mask> could not be set at once')

        elif subset is not None:
            self.set_subset(subset)

        elif mask is not None:
            self.set_mask(mask)

        else:
            self.set_mask([True] * self.n_features)

    def __iter__(self) -> Iterator[Union[int, str]]:
        """
        Returns an iterator over the selected features.
        """
        return iter(self.subset)

    def __len__(self) -> int:
        """
        Returns the number of selected features.
        """
        return self.n_selected

    def __array__(self,
                  *args,
                  **kwargs) -> np.ndarray:
        """
        Returns a numpy array of the selected features.
        """
        return np.array(self.subset, *args, **kwargs)

    def __str__(self) -> str:
        """
        Returns a string representation of the selected features.
        """
        return self.subset.__str__()

    def __repr__(self) -> str:
        """
        Returns a string representation of the FeatureSubset object.
        """
        nm = self.__class__.__name__
        st = self.__str__().replace('\n', '\n ' + ' ' * len(nm))
        return '{}({})'.format(nm, st)

    def __eq__(self,
               other) -> bool:
        """
        Returns True if this FeatureSubset object is equal to the other FeatureSubset object.
        """
        return np.all(self.mask == other.mask)

    def set_subset(self,
                   subset: Sequence[Union[int, str]]) -> 'FeatureSubset':
        """
        Sets the subset of features to the given indices or feature names.

        Args:
            subset: A list or tuple of indices or feature names to be included in the subset.

        Raises:
            AssertionError: If any values in <subset> are not in <features>, or if <subset> values are not unique.

        Returns:
            The FeatureSubset object with the new subset.
        """
        msg = 'Not all <subset> values are in <features>'
        assert np.isin(subset, self.features).all(), msg

        msg = 'All <subset> values must be unique'
        assert len(set(subset)) == len(subset), msg

        self.set_mask(np.isin(self.features, subset))

        return self

    def set_mask(self,
                 mask: np.ndarray) -> 'FeatureSubset':
        """
        Set the mask for the selected features.

        Args:
            mask (np.ndarray): A boolean 1D numpy array with the same length as self.features.

        Returns:
            FeatureSelector: A new FeatureSelector object with the updated mask.
        """
        msg = '<mask> length must be the same as <features>'
        assert len(mask) == self.n_features, msg

        self.mask = np.array(mask, dtype=bool)
        self.subset = self.features[self.mask]

        return self

    def sample(self,
               size: Optional[int] = None,
               random_state: Optional[int] = None) -> 'FeatureSubset':
        """
        Sample a subset of features randomly.

        Args:
            size (int): The size of the subset to sample. If None, a random subset with a random size is returned.
            random_state (int): The random seed to use for reproducibility.

        Returns:
            FeatureSelector: A new FeatureSelector object with the randomly selected subset.
        """
        rstate = check_random_state(random_state)

        if size:
            subset = rstate.choice(self.features, size=size, replace=False)
            return self.copy().set_subset(subset)

        else:
            mask = rstate.randint(0, 2, size=self.n_features, dtype=bool)
            return self.copy().set_mask(mask)

    def remove(self,
               *features,
               copy: bool = True) -> 'FeatureSubset':
        """
        Remove the specified features from the selected subset.

        Args:
            features (List[str]): The feature names to remove from the subset.
            copy (bool): If True, return a new FeatureSelector object. If False, modify the current object in place.

        Returns:
            FeatureSelector: A new FeatureSelector object with the specified features removed.
        """
        self = self.copy() if copy else self

        msg = 'All elements must be unique'
        assert len(set(features)) == len(features), msg

        msg = 'All elements must be in <subset>'
        assert np.isin(features, self.subset).all(), msg

        mask = np.isin(self.features, features)
        self.set_mask(self.mask ^ mask)

        return self

    def append(self,
               *features,
               copy: bool = True) -> 'FeatureSubset':
        """
        Append the specified features to the selected subset.

        Args:
            features (List[str]): The feature names to add to the subset.
            copy (bool): If True, return a new FeatureSelector object. If False, modify the current object in place.

        Returns:
            FeatureSelector: A new FeatureSelector object with the specified features appended.
        """
        self = self.copy() if copy else self

        msg = 'All elements must be unique'
        assert len(set(features)) == len(features), msg

        msg = 'All elements must be in <features>'
        assert np.isin(features, self.features).all(), msg

        msg = 'Some elements already in <subset>'
        assert not np.isin(features, self.subset).any(), msg

        self.set_subset(np.append(self.subset, features))

        return self

    def copy(self) -> 'FeatureSubset':
        """
        Return a copy of the current FeatureSelector object.

        Returns:
            FeatureSelector: A copy of the current FeatureSelector object.
        """
        return copy(self)

    @property
    def n_features(self) -> int:
        """
        Return the number of features in the original feature set.

        Returns:
            int: The number of features in the original feature set.
        """
        return len(self.features)

    @property
    def n_selected(self) -> int:
        """
        Return the number of features in the selected subset.

        Returns:
            int: The number of features in the selected subset.
        """
        return len(self.subset)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Return the shape of the selected subset as a tuple.

        Returns:
            tuple: The shape of the selected subset as a tuple.
        """
        return (self.n_selected,)
