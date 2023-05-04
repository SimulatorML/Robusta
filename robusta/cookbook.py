from robusta.preprocessing.category import *
from robusta.preprocessing.numeric import *
from robusta.preprocessing.base import *
from robusta.pipeline import *




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
