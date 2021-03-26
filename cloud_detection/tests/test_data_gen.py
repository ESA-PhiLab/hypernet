""" Tests for data generator functions. """

import pytest
import numpy as np

from cloud_detection.data_gen import strip_nir


class TestStripNir:
    @pytest.mark.parametrize(
        "in_, true_out",
        [
            (
                np.array(
                    [
                        [[0, 1, 2, 3], [4, 5, 2, 1]],
                        [[0, 3, 1, 2], [0, 4, 3, 1]],
                        [[0, 0, 9, 2], [1, 0, 1, 0]],
                    ]
                ),
                np.array(
                    [
                        [[0, 1, 2], [4, 5, 2]],
                        [[0, 3, 1], [0, 4, 3]],
                        [[0, 0, 9], [1, 0, 1]],
                    ]
                ),
            ),
        ],
    )
    def test_strip_nir(self, in_, true_out):
        np.testing.assert_array_equal(strip_nir(in_), true_out)
