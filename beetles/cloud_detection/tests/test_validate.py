"""
Tests for validation functions.

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

from cloud_detection.validate import (
    find_nearest,
    datagen_to_gt_array,
    find_best_thr
)
from cloud_detection.data_gen import DG_38Cloud, DG_L8CCA


PATH_38CLOUD = Path("datasets/clouds/38-Cloud/38-Cloud_training")
PATH_L8CCA = Path("datasets/clouds/Landsat-Cloud-Cover-Assessment-Validation-Data-Partial")
FILES_38CLOUD = [
    {"red": PATH_38CLOUD / "train_red" / "red_patch_89_5_by_5_LC08_L1TP_034034_20160520_20170223_01_T1.TIF",
     "green": PATH_38CLOUD / "train_green" / "green_patch_89_5_by_5_LC08_L1TP_034034_20160520_20170223_01_T1.TIF",
     "blue": PATH_38CLOUD / "train_blue" / "blue_patch_89_5_by_5_LC08_L1TP_034034_20160520_20170223_01_T1.TIF",
     "nir": PATH_38CLOUD / "train_nir" / "nir_patch_89_5_by_5_LC08_L1TP_034034_20160520_20170223_01_T1.TIF",
     "gt": PATH_38CLOUD / "train_gt" / "gt_patch_89_5_by_5_LC08_L1TP_034034_20160520_20170223_01_T1.TIF"},

    {"red": PATH_38CLOUD / "train_red" / "red_patch_1_1_by_1_LC08_L1TP_002054_20160520_20170324_01_T1.TIF",
     "green": PATH_38CLOUD / "train_green" / "green_patch_1_1_by_1_LC08_L1TP_002054_20160520_20170324_01_T1.TIF",
     "blue": PATH_38CLOUD / "train_blue" / "blue_patch_1_1_by_1_LC08_L1TP_002054_20160520_20170324_01_T1.TIF",
     "nir": PATH_38CLOUD / "train_nir" / "nir_patch_1_1_by_1_LC08_L1TP_002054_20160520_20170324_01_T1.TIF",
     "gt": PATH_38CLOUD / "train_gt" / "gt_patch_1_1_by_1_LC08_L1TP_002054_20160520_20170324_01_T1.TIF"},

    {"red": PATH_38CLOUD / "train_red" / "red_patch_1_1_by_1_LC08_L1TP_011002_20160620_20170323_01_T1.TIF",
     "green": PATH_38CLOUD / "train_green" / "green_patch_1_1_by_1_LC08_L1TP_011002_20160620_20170323_01_T1.TIF",
     "blue": PATH_38CLOUD / "train_blue" / "blue_patch_1_1_by_1_LC08_L1TP_011002_20160620_20170323_01_T1.TIF",
     "nir": PATH_38CLOUD / "train_nir" / "nir_patch_1_1_by_1_LC08_L1TP_011002_20160620_20170323_01_T1.TIF",
     "gt": PATH_38CLOUD / "train_gt" / "gt_patch_1_1_by_1_LC08_L1TP_011002_20160620_20170323_01_T1.TIF"},
    ]
IMG_PATHS_L8CCA = [
    PATH_L8CCA / "Barren" / "LC81640502013179LGN01"
]


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


class TestDatagenToGTArray:
    def test_datagen_to_gt_array(self):
        data_38Cloud = DG_38Cloud(
            files=FILES_38CLOUD,
            batch_size=1,
            shuffle=False
            )
        gt_38Cloud = datagen_to_gt_array(data_38Cloud)
        assert gt_38Cloud.shape == (3, 384, 384, 1)
        np.testing.assert_array_equal(np.unique(gt_38Cloud), [0, 1])
        data_L8CCA = DG_L8CCA(
            img_paths=IMG_PATHS_L8CCA,
            batch_size=1,
            with_gt=True,
            patch_size=100,
            bands=(4, 3),
            bands_names=("red", "green"),
            shuffle=False
            )
        gt_L8CCA = datagen_to_gt_array(data_L8CCA)
        assert gt_L8CCA.shape == (5928, 100, 100, 1)
        np.testing.assert_array_equal(np.unique(gt_L8CCA), [0, 1])


class TestFindBestThr:
    @pytest.mark.parametrize(
        "fpr, tpr, thr, expected",
        [
            (
                np.array([0, 0.1, 0.2]),
                np.array([0, 1, 1]),
                np.array([2, 3, 5]),
                3
            ),
            (
                np.array([0, 10, 0]),
                np.array([0.9, 20, 1]),
                np.array([-2, -1, -0.4]),
                -0.4
            )
        ],
    )
    def test_find_best_thr(self, fpr, tpr, thr, expected):
        assert find_best_thr(fpr=fpr, tpr=tpr, thr=thr) == expected
