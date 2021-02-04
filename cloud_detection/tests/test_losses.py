import pytest
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.debugging import assert_near
from tensorflow import errors, enable_eager_execution

from cloud_detection.losses import (
    Jaccard_index_loss,
    Jaccard_index_metric,
    Dice_coef_metric,
    recall,
    precision,
    specificity,
    f1_score,
)

enable_eager_execution()


class TestJaccardIndexLoss:
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (
                K.constant([[[1, 1], [1, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0),
            ),
            (
                K.constant([[[0, 1], [1, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0.25),
            ),
            (
                K.constant([[[0, 1], [0, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0.5),
            ),
            (
                K.constant([[[0, 0], [0, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0.75),
            ),
            (
                K.constant([[[0, 0], [0, 0]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(1),
            ),
            (
                K.constant([[[1, 1], [0, 0]]]),
                K.constant([[[0.25, 0.25], [0, 0]]]),
                K.constant(0.75),
            ),
            (
                K.constant([[[1, 1], [0, 0]]]),
                K.constant([[[0.75, 0.75], [0, 0]]]),
                K.constant(0.25),
            ),
        ],
    )
    def test_jaccard_index_loss(self, y_true, y_pred, expected):
        jaccard = Jaccard_index_loss()
        try:
            assert_near(jaccard(y_true, y_pred), expected)
        except errors.InvalidArgumentError as e:
            pytest.fail(e.message, pytrace=False)


class TestJaccardIndexMetric:
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (
                K.constant([[[1, 1], [1, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(1),
            ),
            (
                K.constant([[[0, 1], [1, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0.75),
            ),
            (
                K.constant([[[0, 1], [0, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0.5),
            ),
            (
                K.constant([[[0, 0], [0, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0.25),
            ),
            (
                K.constant([[[0, 0], [0, 0]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0),
            ),
            (
                K.constant([[[1, 1], [0, 0]]]),
                K.constant([[[0.25, 0.25], [0, 0]]]),
                K.constant(0),
            ),
            (
                K.constant([[[1, 1], [0, 0]]]),
                K.constant([[[0.75, 0.75], [0, 0]]]),
                K.constant(1),
            ),
        ],
    )
    def test_jaccard_index_metric(self, y_true, y_pred, expected):
        jaccard = Jaccard_index_metric()
        try:
            assert_near(jaccard(y_true, y_pred), expected)
        except errors.InvalidArgumentError as e:
            pytest.fail(e.message, pytrace=False)


class TestDiceCoefMetric:
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (
                K.constant([[[1, 1], [1, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(1),
            ),
            (
                K.constant([[[0, 1], [1, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0.85714286),
            ),
            (
                K.constant([[[0, 1], [0, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0.66666666),
            ),
            (
                K.constant([[[0, 0], [0, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0.4),
            ),
            (
                K.constant([[[0, 0], [0, 0]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0),
            ),
            (
                K.constant([[[1, 1], [0, 0]]]),
                K.constant([[[0.25, 0.25], [0, 0]]]),
                K.constant(0),
            ),
            (
                K.constant([[[1, 1], [0, 0]]]),
                K.constant([[[0.75, 0.75], [0, 0]]]),
                K.constant(1),
            ),
        ],
    )
    def test_dice_coef_metric(self, y_true, y_pred, expected):
        dice = Dice_coef_metric()
        try:
            assert_near(dice(y_true, y_pred), expected)
        except errors.InvalidArgumentError as e:
            pytest.fail(e.message, pytrace=False)


class TestRecall:
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (
                K.constant([[[1, 1], [1, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(1),
            ),
            (
                K.constant([[[0, 1], [1, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(1),
            ),
            (
                K.constant([[[0, 1], [0, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(1),
            ),
            (
                K.constant([[[0, 0], [0, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(1),
            ),
            (
                K.constant([[[0, 0], [0, 0]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0),
            ),
            (
                K.constant([[[1, 1], [0, 0]]]),
                K.constant([[[0.25, 0.25], [0, 0]]]),
                K.constant(0),
            ),
            (
                K.constant([[[1, 1], [0, 0]]]),
                K.constant([[[0.75, 0.75], [0, 0]]]),
                K.constant(1),
            ),
        ],
    )
    def test_recall(self, y_true, y_pred, expected):
        try:
            assert_near(recall(y_true, y_pred), expected)
        except errors.InvalidArgumentError as e:
            pytest.fail(e.message, pytrace=False)


class TestPrecision:
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (
                K.constant([[[1, 1], [1, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(1),
            ),
            (
                K.constant([[[0, 1], [1, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0.75),
            ),
            (
                K.constant([[[0, 1], [0, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0.5),
            ),
            (
                K.constant([[[0, 0], [0, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0.25),
            ),
            (
                K.constant([[[0, 0], [0, 0]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0),
            ),
            (
                K.constant([[[1, 1], [0, 0]]]),
                K.constant([[[0.25, 0.25], [0, 0]]]),
                K.constant(0),
            ),
            (
                K.constant([[[1, 1], [0, 0]]]),
                K.constant([[[0.75, 0.75], [0, 0]]]),
                K.constant(1),
            ),
        ],
    )
    def test_precision(self, y_true, y_pred, expected):
        try:
            assert_near(precision(y_true, y_pred), expected)
        except errors.InvalidArgumentError as e:
            pytest.fail(e.message, pytrace=False)


class TestSpecificity:
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (
                K.constant([[[1, 1], [1, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0),
            ),
            (
                K.constant([[[0, 1], [1, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0),
            ),
            (
                K.constant([[[0, 1], [0, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0),
            ),
            (
                K.constant([[[0, 0], [0, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0),
            ),
            (
                K.constant([[[0, 0], [0, 0]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0),
            ),
            (
                K.constant([[[1, 1], [0, 0]]]),
                K.constant([[[0.25, 0.25], [0, 0]]]),
                K.constant(1),
            ),
            (
                K.constant([[[1, 1], [0, 0]]]),
                K.constant([[[0.75, 0.75], [0, 0]]]),
                K.constant(1),
            ),
        ],
    )
    def test_specificity(self, y_true, y_pred, expected):
        try:
            assert_near(specificity(y_true, y_pred), expected)
        except errors.InvalidArgumentError as e:
            pytest.fail(e.message, pytrace=False)


class TestF1Score:
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            (
                K.constant([[[1, 1], [1, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(1),
            ),
            (
                K.constant([[[0, 1], [1, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0.85714286),
            ),
            (
                K.constant([[[0, 1], [0, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0.66666666),
            ),
            (
                K.constant([[[0, 0], [0, 1]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0.4),
            ),
            (
                K.constant([[[0, 0], [0, 0]]]),
                K.constant([[[1, 1], [1, 1]]]),
                K.constant(0),
            ),
            (
                K.constant([[[1, 1], [0, 0]]]),
                K.constant([[[0.25, 0.25], [0, 0]]]),
                K.constant(0),
            ),
            (
                K.constant([[[1, 1], [0, 0]]]),
                K.constant([[[0.75, 0.75], [0, 0]]]),
                K.constant(1),
            ),
        ],
    )
    def test_f1_score(self, y_true, y_pred, expected):
        try:
            assert_near(f1_score(y_true, y_pred), expected)
        except errors.InvalidArgumentError as e:
            pytest.fail(e.message, pytrace=False)
