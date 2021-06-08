"""
Tests for panchromatic thresholding functions.

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

from cloud_detection.scripts.panchromatic_thresholding import \
    ThresholdingClassifier


class TestFit:
    @pytest.mark.parametrize(
        "thr_prop, X, min_expected, max_expected, thr_expected",
        [
            (0.5, np.array([1, 1, 4, 2]).reshape(2, 2), 1, 4, 2.5),
            (0.3, np.array([0, 1, 3]).reshape(1, 3), 1, 3, 1.6)
        ],
    )
    def test_fit(self, thr_prop, X, min_expected, max_expected, thr_expected):
        thr_classifier = ThresholdingClassifier(thr_prop=thr_prop)
        thr_classifier.fit(X)
        assert thr_classifier.min == min_expected
        assert thr_classifier.max == max_expected
        assert thr_classifier.thr == thr_expected


class TestPredict:
    @pytest.mark.parametrize(
        "thr_prop, X, expected",
        [
            (0.5, np.array([1, 1, 4, 2.5]).reshape(2, 2),
             np.array([0, 0, 1, 1]).reshape(2, 2)),
            (0.3, np.array([0, 1, 3]).reshape(1, 3),
             np.array([0, 0, 1]).reshape(1, 3))
        ],
    )
    def test_predict(self, thr_prop, X, expected):
        thr_classifier = ThresholdingClassifier(thr_prop=thr_prop)
        thr_classifier.fit(X)
        mask = thr_classifier.predict(X)
        assert X.shape == mask.shape
        np.testing.assert_array_equal(mask, expected)
