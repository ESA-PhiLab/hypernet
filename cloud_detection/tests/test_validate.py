import pytest
import numpy as np

from cloud_detection.validate import find_nearest


class TestFindNearest:
    @pytest.mark.parametrize(
        "array, value, true_out",
        [
            (np.array([[0, 0], [1, 0]]), 1, 2),
            (np.array([[[0, 2, 0], [0, 1, 3.5]], [[1, 0, 10], [4, 0, 0]]]),
             3.8, 9),
        ],
    )
    def test_find_nearest(self, array, value, true_out):
        assert find_nearest(array, value) == true_out
