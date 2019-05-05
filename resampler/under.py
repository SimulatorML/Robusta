from imblearn import under_sampling

from .base import make_sampler


ClusterCentroids = make_sampler(under_sampling.ClusterCentroids)
RUS = make_sampler(under_sampling.RandomUnderSampler)
IHT = make_sampler(under_sampling.InstanceHardnessThreshold)
NearMiss = make_sampler(under_sampling.NearMiss)
TomekLinks = make_sampler(under_sampling.TomekLinks)
ENN = make_sampler(under_sampling.EditedNearestNeighbours)
RENN = make_sampler(under_sampling.RepeatedEditedNearestNeighbours)
AllKNN = make_sampler(under_sampling.AllKNN)
OSS = make_sampler(under_sampling.OneSidedSelection)
CNN = make_sampler(under_sampling.CondensedNearestNeighbour)
NCR = make_sampler(under_sampling.NeighbourhoodCleaningRule)
