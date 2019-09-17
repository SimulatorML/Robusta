import numpy as np
import pandas as pd

from .preprocessing.category import *
from .preprocessing.numeric import *
from .preprocessing.base import *
from .pipeline import *




mem_reduce_pipe = FeatureUnion([
    ('numeric', make_pipeline(
        TypeSelector(np.number),
        NumDowncast(),
    )),
    ('category', make_pipeline(
        TypeSelector('object'),
    )),
])



prep_pipe = make_pipeline(
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
    ])
)