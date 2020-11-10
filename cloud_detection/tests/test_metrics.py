import pytest
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.debugging import assert_near
from tensorflow import errors

from cloud_detection.losses import Jaccard_index_loss, Jaccard_index_metric


@pytest.mark.parametrize(
    'y_true, y_pred, expected',
    [
        (K.constant([[ [1, 1], [1, 1] ]]), K.constant([[ [1, 1], [1, 1] ]]), K.constant(0)),
        (K.constant([[ [0, 1], [1, 1] ]]), K.constant([[ [1, 1], [1, 1] ]]), K.constant(0.25)),
        (K.constant([[ [0, 1], [0, 1] ]]), K.constant([[ [1, 1], [1, 1] ]]), K.constant(0.5)),
        (K.constant([[ [0, 0], [0, 1] ]]), K.constant([[ [1, 1], [1, 1] ]]), K.constant(0.75)),
        (K.constant([[ [0, 0], [0, 0] ]]), K.constant([[ [1, 1], [1, 1] ]]), K.constant(1)),
        (K.constant([[ [1, 1], [0, 0] ]]), K.constant([[ [0.25, 0.25], [0, 0] ]]), K.constant(0.75)),
        (K.constant([[ [1, 1], [0, 0] ]]), K.constant([[ [0.75, 0.75], [0, 0] ]]), K.constant(0.25)),
    ]
)
def test_jaccard_index_loss(y_true, y_pred, expected):
    jaccard = Jaccard_index_loss()
    try:
        assert_near(jaccard(y_true, y_pred), expected)
    except errors.InvalidArgumentError as e:
        pytest.fail(e.message, pytrace=False)


@pytest.mark.parametrize(
    'y_true, y_pred, expected',
    [
        (K.constant([[ [1, 1], [1, 1] ]]), K.constant([[ [1, 1], [1, 1] ]]), K.constant(1)),
        (K.constant([[ [0, 1], [1, 1] ]]), K.constant([[ [1, 1], [1, 1] ]]), K.constant(0.75)),
        (K.constant([[ [0, 1], [0, 1] ]]), K.constant([[ [1, 1], [1, 1] ]]), K.constant(0.5)),
        (K.constant([[ [0, 0], [0, 1] ]]), K.constant([[ [1, 1], [1, 1] ]]), K.constant(0.25)),
        (K.constant([[ [0, 0], [0, 0] ]]), K.constant([[ [1, 1], [1, 1] ]]), K.constant(0)),
        (K.constant([[ [1, 1], [0, 0] ]]), K.constant([[ [0.25, 0.25], [0, 0] ]]), K.constant(0)),
        (K.constant([[ [1, 1], [0, 0] ]]), K.constant([[ [0.75, 0.75], [0, 0] ]]), K.constant(1)),
    ]
)
def test_jaccard_index_metric(y_true, y_pred, expected):
    jaccard = Jaccard_index_metric()
    try:
        assert_near(jaccard(y_true, y_pred), expected)
    except errors.InvalidArgumentError as e:
        pytest.fail(e.message, pytrace=False)

