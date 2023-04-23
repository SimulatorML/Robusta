import numpy as np

from robusta.metrics import paired_radian_distances


def test_paired_radian_distances():
    X = np.array([[37.774929, -122.419416], [40.712776, -74.005974], [51.507351, -0.127758]])
    Y = np.array([[37.774929, -122.419416], [52.520008, 13.404954], [35.689487, 139.691711]])
    expected_distances = np.array([0., 6387.00506158, 9561.71922372])

    distances = paired_radian_distances(X, Y)

    assert np.allclose(distances, expected_distances)