from sklearn.utils.random import check_random_state

import pandas as pd
import numpy as np



class FeatureSubset:

    def __init__(self, features, subset=None, mask=None, group=False):

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
            self.set_mask([True]*self.n_features)


    def __iter__(self):
        return iter(self.subset)


    def __len__(self):
        return self.n_selected


    def __array__(self, *args, **kwargs):
        return np.array(self.subset, *args, **kwargs)


    def __str__(self):
        return self.subset.__str__()


    def __repr__(self):
        nm = self.__class__.__name__
        st = self.__str__().replace('\n', '\n ' + ' '*len(nm))
        return '{}({})'.format(nm, st)


    def __eq__(self, other):
        return np.all(self.mask == other.mask)


    def set_subset(self, subset):

        msg = 'Not all <subset> values are in <features>'
        assert np.isin(subset, self.features).all(), msg

        msg = 'All <subset> values must be unique'
        assert len(set(subset)) == len(subset), msg

        self.set_mask(np.isin(self.features, subset))

        return self


    def set_mask(self, mask):

        msg = '<mask> length must be the same as <features>'
        assert len(mask) == self.n_features, msg

        self.mask = np.array(mask, dtype=bool)
        self.subset = self.features[self.mask]

        return self


    def sample(self, size=None, random_state=None):

        rstate = check_random_state(random_state)

        if size:
            subset = rstate.choice(self.features, size=size, replace=False)
            return self.copy().set_subset(subset)

        else:
            mask = rstate.randint(0, 2, size=self.n_features, dtype=bool)
            return self.copy().set_mask(mask)


    def remove(self, *features, copy=True):

        self = self.copy() if copy else self

        msg = 'All elements must be unique'
        assert len(set(features)) == len(features), msg

        msg = 'All elements must be in <subset>'
        assert np.isin(features, self.subset).all(), msg

        mask = np.isin(self.features, features)
        self.set_mask(self.mask ^ mask)

        return self


    def append(self, *features, copy=True):

        self = self.copy() if copy else self

        msg = 'All elements must be unique'
        assert len(set(features)) == len(features), msg

        msg = 'All elements must be in <features>'
        assert np.isin(features, self.features).all(), msg

        msg = 'Some elements already in <subset>'
        assert not np.isin(features, self.subset).any(), msg

        self.set_subset(np.append(self.subset, features))

        return self


    def copy(self):
        return copy(self)

    @property
    def n_features(self):
        return len(self.features)

    @property
    def n_selected(self):
        return len(self.subset)

    @property
    def shape(self):
        return (self.n_selected, )
