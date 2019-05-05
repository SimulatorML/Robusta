from imblearn import combine

from .base import make_sampler


SMOTEENN = make_sampler(combine.SMOTEENN)
SMOTETomek = make_sampler(combine.SMOTETomek)
