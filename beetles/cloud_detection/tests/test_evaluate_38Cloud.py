"""
Tests for 38Cloud evaluation functions.

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

from cloud_detection.evaluate_38Cloud import (
    get_full_scene_img,
    get_img_pred_shape,
    load_img_gt
)


PATH_38CLOUD = Path("datasets/clouds/38-Cloud/38-Cloud_training")
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
IMG_ID = "LC08_L1TP_002053_20160520_20170324_01_T1.TIF"
GT_ID = "edited_corrected_gts_LC08_L1TP_002053_20160520_20170324_01_T1.TIF"


class TestGetFullSceneImg:
    def test_get_full_scene_img(self):
        img = get_full_scene_img(
            path=PATH_38CLOUD / "Natural_False_Color",
            img_id=IMG_ID
            )
        assert img.shape == (7761, 7601, 3)
        assert np.isclose(np.min(img), 0)
        assert np.isclose(np.max(img), 1)


class TestGetImgPredShape:
    @pytest.mark.parametrize(
        "patch_size, expected",
        [
            (100, (100*5, 100*5, 1)),
            (384, (384*5, 384*5, 1))
        ],
    )
    def test_get_img_pred_shape(self, patch_size, expected):
        s = get_img_pred_shape(files=FILES_38CLOUD, patch_size=patch_size)
        assert s == expected


class TestLoadImgGT:
    def test_load_img_gt(self):
        gt = load_img_gt(
            path=PATH_38CLOUD / "Entire_scene_gts",
            fname=GT_ID
            )
        assert gt.shape == (7761, 7601, 1)
        np.testing.assert_array_equal(np.unique(gt), [0, 1])
