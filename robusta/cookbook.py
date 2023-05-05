from pipeline._pipeline import make_pipeline
from preprocessing.category import *
from preprocessing.numeric import *
from preprocessing.base import *
from pipeline import *

mem_reduce_pipe = FeatureUnion([
    ('numeric', make_pipeline(
        TypeSelector(np.number),
        DowncastTransformer(),
    )),
    ('category', make_pipeline(
        TypeSelector('object'),
    )),
])

prep_pipe = make_pipeline(
    FeatureUnion([
        ("numeric", make_pipeline(
            TypeSelector(np.number),
            SimpleImputer("median"),
            GaussRankTransformer(),
            ColumnRenamer(prefix='gr_'),
        )),
        ("category", make_pipeline(
            TypeSelector("object"),
            LabelEncoder(),
            ColumnRenamer(prefix='le_'),
        )),
    ])
)
