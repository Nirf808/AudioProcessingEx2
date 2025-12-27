import numpy as np
from DTW import calc_dtw


def test_dtw_identical():
    # Reshape 1D arrays to (N, 1)
    a = np.array([1, 2, 3], dtype=float).reshape(-1, 1)
    b = np.array([1, 2, 3], dtype=float).reshape(-1, 1)

    expected_distance = 0.0
    # Use np.isclose for float comparison
    assert np.isclose(calc_dtw(a, b), expected_distance)


def test_dtw_stretched():
    a = np.array([1, 2, 3], dtype=float).reshape(-1, 1)
    b = np.array([1, 1, 2, 2, 3, 3], dtype=float).reshape(-1, 1)

    expected_distance = 0.0
    assert np.isclose(calc_dtw(a, b), expected_distance)


def test_dtw_shift():
    # These are already 2D, but ensure they are (N, Features)
    # Your original [[1, 3]] is 1 sample with 2 features.
    # If you meant a sequence of 2 points, use .reshape(-1, 1)
    a = np.array([1, 3], dtype=float).reshape(-1, 1)
    b = np.array([2, 2], dtype=float).reshape(-1, 1)

    # |1-2| + |3-2| = 1 + 1 = 2
    expected_distance = 2.0
    assert np.isclose(calc_dtw(a, b), expected_distance)


if __name__ == '__main__':
    test_dtw_identical()
    test_dtw_stretched()
    test_dtw_shift()
    print("All tests passed!")
