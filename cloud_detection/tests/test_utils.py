"""
Tests for utils functions.

If you plan on using this implementation, please cite our work:
@INPROCEEDINGS{Grabowski2021IGARSS,
author={Grabowski, Bartosz and Ziaja, Maciej and Kawulok, Michal
and Nalepa, Jakub},
booktitle={IGARSS 2021 - 2021 IEEE International Geoscience
and Remote Sensing Symposium},
title={Towards Robust Cloud Detection in
Satellite Images Using U-Nets},
year={2021},
note={in press}}
"""

import pytest
import numpy as np
from pathlib import Path
# from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy

from cloud_detection.utils import (
    true_positives,
    false_positives,
    false_negatives,
    overlay_mask,
    pad,
    unpad,
    make_paths,
    strip_nir,
    open_as_array,
    load_38cloud_gt,
    load_l8cca_gt,
    load_image_paths,
    combine_channel_files,
    build_paths,
    # get_metrics_tf
)


PATH_38CLOUD = Path("datasets/clouds/38-Cloud/38-Cloud_training")
PATH_L8CCA = Path("datasets/clouds/Landsat-Cloud-Cover-Assessment-Validation-Data-Partial")
CHANNEL_FILES_38CLOUD = {
     "red": PATH_38CLOUD / "train_red" / "red_patch_89_5_by_5_LC08_L1TP_034034_20160520_20170223_01_T1.TIF",
     "green": PATH_38CLOUD / "train_green" / "green_patch_89_5_by_5_LC08_L1TP_034034_20160520_20170223_01_T1.TIF",
     "blue": PATH_38CLOUD / "train_blue" / "blue_patch_89_5_by_5_LC08_L1TP_034034_20160520_20170223_01_T1.TIF",
     "nir": PATH_38CLOUD / "train_nir" / "nir_patch_89_5_by_5_LC08_L1TP_034034_20160520_20170223_01_T1.TIF",
     "gt": PATH_38CLOUD / "train_gt" / "gt_patch_89_5_by_5_LC08_L1TP_034034_20160520_20170223_01_T1.TIF"
     }
IMG_ID_38CLOUD = "LC08_L1TP_002053_20160520_20170324_01_T1"
PATCHES_PATH_38CLOUD = PATH_38CLOUD / ".." / "training_patches_38-cloud_nonempty.csv"
IMG_PATH_L8CCA = PATH_L8CCA / "Barren" / "LC81640502013179LGN01"


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
            ("123", None, (Path("123"), None))
        ],
    )
    def test_make_paths(self, path1, path2, paths):
        assert make_paths(path1, path2) == paths


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


class TestOpenAsArray:
    @pytest.mark.parametrize(
        "size, normalize, standardize",
        [
            ((100, 150), True, False),
            (None, False, False),
            (None, False, True),
            ((384, 384), True, True)
        ],
    )
    def test_open_as_array(self, size, normalize, standardize):
        img = open_as_array(
            channel_files=CHANNEL_FILES_38CLOUD,
            size=size,
            normalize=normalize,
            standardize=standardize
            )
        if size is not None:
            assert img.shape == (*size, 4)
        else:
            assert img.shape == (384, 384, 4)
        if normalize and not standardize:
            assert np.min(img) >= 0
            assert np.max(img) <= 1
        if standardize:
            for b in range(4):
                assert np.isclose(np.mean(img[:, :, b]), 0)
                assert np.isclose(np.std(img[:, :, b]), 1)


class TestLoad38CloudGT:
    def test_load_38cloud_gt(self):
        gt = load_38cloud_gt(
            channel_files=CHANNEL_FILES_38CLOUD
            )
        assert gt.shape == (384, 384, 1)
        np.testing.assert_array_equal(np.unique(gt), [0, 1])


class TestLoadL8CCAGT:
    def test_load_l8cca_gt(self):
        gt = load_l8cca_gt(
            path=IMG_PATH_L8CCA
            )
        assert gt.shape == (7741, 7551, 1)
        np.testing.assert_array_equal(np.unique(gt), [0, 1])


class TestLoadImagePaths:
    @pytest.mark.parametrize(
        "patches_path, split_ratios, shuffle, expected",
        [
            (None, (1.,), True, 420),
            (PATCHES_PATH_38CLOUD, (1.,), False, 288),
            (None, (0.5, 0.38, 0.12), False, 420),
            (None, (0.1, 0.1, 0.8), True, 420)
        ],
    )
    def test_load_image_paths(
        self, patches_path, split_ratios, shuffle, expected
    ):
        img_paths = load_image_paths(
            base_path=PATH_38CLOUD,
            patches_path=patches_path,
            split_ratios=split_ratios,
            shuffle=False,
            img_id=IMG_ID_38CLOUD
            )
        if shuffle:
            img_paths_shuffled_splitted = load_image_paths(
                base_path=PATH_38CLOUD,
                patches_path=patches_path,
                split_ratios=split_ratios,
                shuffle=True,
                img_id=IMG_ID_38CLOUD,
                seed=100
                )
            img_paths_shuffled = []
            for split in img_paths_shuffled_splitted:
                img_paths_shuffled.extend(list(split))
            img_paths_unshuffled = []
            for split in img_paths:
                img_paths_unshuffled.extend(list(split))
            saved_seed = np.random.get_state()
            np.random.seed(100)
            np.random.shuffle(img_paths_unshuffled)
            np.random.set_state(saved_seed)
            np.testing.assert_array_equal(
                img_paths_shuffled,
                img_paths_unshuffled
                )
        assert set(img_paths[0][0].keys()) == set(("red", "green", "blue",
                                                   "nir", "gt"))
        assert len(img_paths) == len(split_ratios)
        paths_sum = 0
        for i, split in enumerate(img_paths):
            assert abs(len(split) - int(split_ratios[i] * expected)) <= 1
            paths_sum += len(split)
        assert paths_sum == expected


class TestCombineChannelFiles:
    def test_combine_channel_files(self):
        img_paths = combine_channel_files(
            red_file=CHANNEL_FILES_38CLOUD["red"]
            )
        assert set(img_paths.keys()) == set(("red", "green", "blue",
                                             "nir", "gt"))


class TestBuildPaths:
    @pytest.mark.parametrize(
        "patches_path, expected",
        [
            (None, 420),
            (PATCHES_PATH_38CLOUD, 288)
        ],
    )
    def test_build_paths(self, patches_path, expected):
        img_paths = build_paths(
            base_path=PATH_38CLOUD,
            patches_path=patches_path,
            img_id=IMG_ID_38CLOUD
            )
        assert len(img_paths) == expected
        assert set(img_paths[0].keys()) == set(("red", "green", "blue",
                                                "nir", "gt"))


# # Won't work when run together with losses.py tests because of
# # eager execution.
# class TestGetMetricsTF:
#     @pytest.mark.parametrize(
#         "gt, pred, metric_fns, expected",
#         [
#             (np.array([0, 1, 1, 0]).reshape(1, 2, 2, 1),
#              np.array([1, 1, 1, 0]).reshape(1, 2, 2, 1),
#              [binary_accuracy],
#              {"binary_accuracy": 0.75}),
#         ],
#     )
#     def test_get_metrics_tf(self, gt, pred, metric_fns, expected):
#         assert get_metrics_tf(
#             gt=gt,
#             pred=pred,
#             metric_fns=metric_fns
#             ) == expected
