from imblearn import over_sampling

from .base import make_sampler


ADASYN = make_sampler(over_sampling.ADASYN)
SMOTE = make_sampler(over_sampling.SMOTE)
SVMSMOTE = make_sampler(over_sampling.SVMSMOTE)
SMOTENC = make_sampler(over_sampling.SMOTENC)
BSMOTE = make_sampler(over_sampling.BorderlineSMOTE)
ROS = make_sampler(over_sampling.RandomOverSampler)
