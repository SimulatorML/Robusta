import numpy as np
import pandas as pd

from math import atan2, radians

from sklearn.metrics.pairwise import (paired_euclidean_distances,
                                      paired_manhattan_distances,
                                      paired_cosine_distances)

__all__ = [
    'paired_euclidean_distances',
    'paired_manhattan_distances',
    'paired_cosine_distances',
    'paired_radian_distances',
]


def paired_radian_distances(X, Y, radius=6373.0):

    # Convert Degrees to Radians
    X = np.radians(X.values)
    Y = np.radians(Y.values)

    # Latitude & Longitude difference
    latX, lonX = X.T
    latY, lonY = Y.T

    dlat, dlon = latX-latY, lonX-lonY

    # Distance on Sphere
    d = np.sin(dlat/2)**2 + np.cos(latX) * np.cos(latY) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(d), np.sqrt(1 - d))

    return radius * c
