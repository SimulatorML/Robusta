from .paired import *
from .pairwise import *


PAIRED_DISTANCES = {
    'cosine': paired_cosine_distances,
    'euclidean': paired_euclidean_distances,
    'l2': paired_euclidean_distances,
    'l1': paired_manhattan_distances,
    'manhattan': paired_manhattan_distances,
    'cityblock': paired_manhattan_distances,
    'radian': paired_radian_distances,
    'coordinate': paired_radian_distances,
}


def paired_distances(X, Y, metric='euclidean', **kwargs):
    return PAIRED_DISTANCES[metric](X, Y, **kwargs)
