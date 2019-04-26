import numpy as np
import pandas as pd

from ..preprocessing.category import *
from ..preprocessing.numeric import *
from ..preprocessing import *

from ._pipeline import *




prep_pipe = make_pipeline(
    ColumnSelector(columns=features),
    FeatureUnion([
        ("numeric", make_pipeline(
            TypeSelector(np.number),
            Imputer(strategy="median"),
            GaussRank(),
            ColumnRenamer(prefix='gr_'),
        )),
        ("category", make_pipeline(
            TypeSelector("object"),
            LabelEncoder(),
            ColumnRenamer(prefix='le_'),
        )),
        #("boolean", make_pipeline(
        #    TypeSelector("bool"),
        #    Imputer(strategy="most_frequent")
        #)),
    ])
)
