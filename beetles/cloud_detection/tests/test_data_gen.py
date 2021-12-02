"""
Tests for data gen functions.

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

from cloud_detection.data_gen import (
    DG_38Cloud,
    DG_L8CCA
)


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


class TestPerformBalancing:
    @pytest.mark.parametrize(
        "labels, expected",
        [
            ([0, 1, 0], np.array([0, 1, 1, 2])),
            ([1, 0, 1], np.array([0, 1, 1, 2])),
            ([0, 1, 1], np.array([0, 0, 1, 2]))
        ],
    )
    def test__perform_balancing(self, labels, expected):
        data_gen = DG_38Cloud(
            files=FILES_38CLOUD,
            batch_size=1,
            shuffle=False
            )
        data_gen._perform_balancing(labels=labels)
        np.testing.assert_array_equal(data_gen._file_indexes, expected)


class TestGetLabelsForBalancing:
    @pytest.mark.parametrize(
        "min_prop, max_prop, expected",
        [
            (0.01, 0.1, [1, 0, 0]),
            (0.1, 0.9, [0, 0, 0])
        ],
    )
    def test__get_labels_for_balancing(self, min_prop, max_prop, expected):
        data_gen = DG_38Cloud(
            files=FILES_38CLOUD,
            batch_size=1,
            shuffle=False
            )
        labels = data_gen._get_labels_for_balancing(
            min_prop=min_prop,
            max_prop=max_prop
        )
        np.testing.assert_array_equal(labels, expected)


class TestGetLabelsForSnowBalancing:
    @pytest.mark.parametrize(
        "brightness_thr, frequency_thr, expected",
        [
            (0.5, 0.5, [1, 0, 0]),
            (0.5, 0.6, [0, 0, 0]),
            (0.3, 0.7, [1, 0, 0]),
            (0.1, 1.0, [0, 0, 0]),
            (0.9, 0.0, [1, 0, 0]),
        ],
    )
    def test__get_labels_for_snow_balancing(
        self,
        brightness_thr,
        frequency_thr,
        expected
    ):
        data_gen = DG_38Cloud(
            files=FILES_38CLOUD,
            batch_size=1,
            shuffle=False
            )
        labels = data_gen._get_labels_for_snow_balancing(
            brightness_thr=brightness_thr,
            frequency_thr=frequency_thr
        )
        np.testing.assert_array_equal(labels, expected)


class TestDataGeneration:
    @pytest.mark.parametrize(
        "file_indexes_to_gen, with_gt, x_expected, y_expected",
        [
            ([0, 1, 2], True, (3, 384, 384, 4), (3, 384, 384, 1)),
            ([0, 1], False, (2, 384, 384, 4), None)
        ],
    )
    def test__data_generation(
        self,
        file_indexes_to_gen,
        with_gt,
        x_expected,
        y_expected
    ):
        data_gen = DG_38Cloud(
            files=FILES_38CLOUD,
            batch_size=1,
            shuffle=False,
            with_gt=with_gt
            )
        x, y = data_gen._data_generation(
            file_indexes_to_gen=file_indexes_to_gen
        )
        assert x.shape == x_expected
        if with_gt:
            assert y.shape == y_expected
        else:
            assert y == y_expected


class Test38CloudLen:
    @pytest.mark.parametrize(
        "batch_size, expected",
        [
            (1, 3),
            (2, 2),
            (3, 1)
        ],
    )
    def test_38Cloud__len__(
        self,
        batch_size,
        expected
    ):
        data_gen = DG_38Cloud(
            files=FILES_38CLOUD,
            batch_size=batch_size,
            shuffle=False
            )
        assert data_gen.__len__() == expected


class Test38CloudGetItem:
    @pytest.mark.parametrize(
        "batch_size, with_gt, x_expected, y_expected",
        [
            (1, True, (1, 384, 384, 4), (1, 384, 384, 1)),
            (3, False, (3, 384, 384, 4), None)
        ],
    )
    def test_38Cloud__getitem__(
        self,
        batch_size,
        with_gt,
        x_expected,
        y_expected
    ):
        data_gen = DG_38Cloud(
            files=FILES_38CLOUD,
            batch_size=batch_size,
            shuffle=False,
            with_gt=with_gt
            )
        x, y = data_gen.__getitem__(0)
        assert x.shape == x_expected
        if with_gt:
            assert y.shape == y_expected
        else:
            assert y == y_expected


class TestGenerateImgPatches:
    @pytest.mark.parametrize(
        "patch_size, bands, bands_names, expected",
        [
            (100, (4, 3), ("red", "green"), (5928, 100, 100, 2))
        ],
    )
    def test__generate_img_patches(
        self,
        patch_size,
        bands,
        bands_names,
        expected
    ):
        data_gen = DG_L8CCA(
            img_paths=IMG_PATHS_L8CCA,
            batch_size=1,
            patch_size=patch_size,
            bands=bands,
            bands_names=bands_names,
            shuffle=False
            )
        data_gen._generate_img_patches(bands, bands_names)
        assert data_gen.patches.shape == expected


class TestGenerateGTPatches:
    @pytest.mark.parametrize(
        "patch_size, expected",
        [
            (648, (144, 648, 648, 1))
        ],
    )
    def test__generate_gt_patches(
        self,
        patch_size,
        expected
    ):
        data_gen = DG_L8CCA(
            img_paths=IMG_PATHS_L8CCA,
            batch_size=1,
            with_gt=True,
            patch_size=patch_size,
            shuffle=False
            )
        data_gen._generate_gt_patches()
        assert data_gen.gt_patches.shape == expected


class TestPartitionData:
    @pytest.mark.parametrize(
        "data_part, expected",
        [
            ((0.2, 0.8), 0.6)
        ],
    )
    def test__partition_data(
        self,
        data_part,
        expected
    ):
        data_gen = DG_L8CCA(
            img_paths=IMG_PATHS_L8CCA,
            batch_size=1,
            data_part=data_part,
            shuffle=False
            )
        old_patches_num = len(data_gen._patches_indexes)
        data_gen._partition_data()
        new_patches_num = len(data_gen._patches_indexes)
        assert abs(new_patches_num - int(expected * old_patches_num)) <= 2


class TestL8CCALen:
    @pytest.mark.parametrize(
        "batch_size, expected",
        [
            (4, 105)
        ],
    )
    def test_L8CCA__len__(
        self,
        batch_size,
        expected
    ):
        data_gen = DG_L8CCA(
            img_paths=IMG_PATHS_L8CCA,
            batch_size=batch_size,
            shuffle=False
            )
        assert data_gen.__len__() == expected


class TestL8CCAGetItem:
    @pytest.mark.parametrize(
        "batch_size, x_expected, y_expected",
        [
            (2, (2, 384, 384, 4), (2, 384, 384, 1))
        ],
    )
    def test_L8CCA__getitem__(
        self,
        batch_size,
        x_expected,
        y_expected
    ):
        data_gen = DG_L8CCA(
            img_paths=IMG_PATHS_L8CCA,
            batch_size=batch_size,
            with_gt=True,
            shuffle=False
            )
        x, y = data_gen.__getitem__(0)
        assert x.shape == x_expected
        assert y.shape == y_expected
