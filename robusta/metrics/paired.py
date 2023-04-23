from typing import Union

import numpy as np


def paired_radian_distances(X: np.ndarray,
                            Y: np.ndarray,
                            radius: Union[float, int] = 6373.0) -> np.array:
    """
    Calculate the paired distances between two arrays of coordinates in radians.

    Args:
    - X: a numpy array of shape (n, 2) containing n pairs of coordinates (latitude, longitude) in radians.
    - Y: a numpy array of shape (n, 2) containing n pairs of coordinates (latitude, longitude) in radians.
    - radius: a float or integer representing the radius of the sphere used to calculate the distances.
        Defaults to the radius of the Earth in kilometers.

    Returns:
    - A numpy array of shape (n,) containing the distances between the paired coordinates in kilometers.
    """

    # Convert Degrees to Radians
    X = np.radians(X)
    Y = np.radians(Y)

    # Latitude & Longitude difference
    latX, lonX = X.T
    latY, lonY = Y.T

    dlat, dlon = latX - latY, lonX - lonY

    # Distance on Sphere
    d = np.sin(dlat / 2) ** 2 + np.cos(latX) * np.cos(latY) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(d), np.sqrt(1 - d))

    return radius * c
