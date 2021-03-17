import pytest
import numpy as np
from pathlib import Path

from cloud_detection.utils import (
    true_positives,
    false_positives,
    false_negatives,
    overlay_mask,
    pad,
    unpad,
    make_paths,
)


class TestTruePositives:
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])),
            (np.array([0, 1, 1]), np.array([1, 0, 1]), np.array([0, 0, 1])),
            (np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])),
        ],
    )
    def test_true_positives(self, y_true, y_pred, expected):
        np.testing.assert_array_equal(true_positives(y_true, y_pred), expected)


class TestFalsePositives:
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])),
            (np.array([0, 1, 1]), np.array([1, 0, 1]), np.array([1, 0, 0])),
            (np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([0, 0, 0])),
        ],
    )
    def test_false_positives(self, y_true, y_pred, expected):
        np.testing.assert_array_equal(
            false_positives(y_true, y_pred), expected)


class TestFalseNegatives:
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])),
            (np.array([0, 1, 1]), np.array([1, 0, 1]), np.array([0, 1, 0])),
            (np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([0, 0, 0])),
        ],
    )
    def test_false_negatives(self, y_true, y_pred, expected):
        np.testing.assert_array_equal(
            false_negatives(y_true, y_pred), expected)


class TestOverlayMask:
    @pytest.mark.parametrize(
        "image, mask, rgb_color, overlay_intensity, out_true",
        [
            (
                np.array(
                    [
                        [[0.0, 0.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 0.0]],
                    ]
                ),
                np.array([[[1], [0]], [[1], [1]], [[1], [0]]]),
                (0.0, 0.5),
                1,
                np.array(
                    [
                        [[0.0, 0.5], [0.0, 0.0]],
                        [[0.0, 0.5], [0.0, 0.5]],
                        [[0.0, 0.5], [0.0, 0.0]],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [[1.5, 0.0], [0.0, 0.3]],
                        [[1.4, 0.0], [1.0, 0.0]],
                        [[0.5, 0.6], [0.9, 10.0]],
                    ]
                ),
                np.array([[[1], [0]], [[0], [1]], [[1], [0]]]),
                (1.0, 0.5),
                0.5,
                np.array(
                    [
                        [[1.0, 0.25], [0.0, 0.3]],
                        [[1.0, 0.0], [1.0, 0.25]],
                        [[1.0, 0.85], [0.9, 1.0]],
                    ]
                ),
            ),
        ],
    )
    def test_overlay_mask(
            self, image, mask, rgb_color, overlay_intensity, out_true):
        out = overlay_mask(image, mask, rgb_color, overlay_intensity)
        np.testing.assert_array_equal(out, out_true)


class TestPad:
    @pytest.mark.parametrize(
        "img, patch_size, out_true",
        [
            (
                np.array([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]]),
                3,
                np.array(
                    [
                        [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
                        [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    ]
                ),
            ),
            (
                np.array(
                    [[[1, 1, 1], [1, 0.6, 1]], [[1, 1, 10.3], [1, 1, 1]]]),
                4,
                np.array(
                    [
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[0, 0, 0], [1, 1, 1], [1, 0.6, 1], [0, 0, 0]],
                        [[0, 0, 0], [1, 1, 10.3], [1, 1, 1], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    ]
                ),
            ),
            (
                np.array([[[1, 1, 1], [0, 0, 1]], [[1, 1, 1], [1, 1, 0]]]),
                2,
                np.array([[[1, 1, 1], [0, 0, 1]], [[1, 1, 1], [1, 1, 0]]]),
            ),
        ],
    )
    def test_pad(self, img, patch_size, out_true):
        np.testing.assert_array_equal(pad(img, patch_size), out_true)


class TestUnpad:
    @pytest.mark.parametrize(
        "img, gt_shape, out_true",
        [
            (
                np.array(
                    [
                        [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
                        [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    ]
                ),
                (2, 2, 3),
                np.array([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]]),
            ),
            (
                np.array(
                    [
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[0, 0, 0], [1, 1, 1], [1, 0.6, 1], [0, 0, 0]],
                        [[0, 0, 0], [1, 1, 10.3], [1, 1, 1], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    ]
                ),
                (2, 2, 3),
                np.array(
                    [[[1, 1, 1], [1, 0.6, 1]], [[1, 1, 10.3], [1, 1, 1]]]),
            ),
            (
                np.array([[[1, 1, 1], [0, 0, 1]], [[1, 1, 1], [1, 1, 0]]]),
                (2, 2, 3),
                np.array([[[1, 1, 1], [0, 0, 1]], [[1, 1, 1], [1, 1, 0]]]),
            ),
        ],
    )
    def test_unpad(self, img, gt_shape, out_true):
        np.testing.assert_array_equal(unpad(img, gt_shape), out_true)


class TestMakePaths:
    @pytest.mark.parametrize(
        "path1, path2, paths",
        [
            ("ab/cd", "efg/hij", (Path("ab/cd"), Path("efg/hij"))),
        ],
    )
    def test_make_paths(self, path1, path2, paths):
        assert make_paths(path1, path2) == paths
