"""Custom loss functions and metrics for segmentation."""

from collections.abc import Callable

import tensorflow.keras.backend as K
import numpy as np

# Import all other metrics used by the model
# to enable testing it.
# noqa # pylint: disable=unused-import
from tensorflow.keras.metrics import binary_crossentropy, binary_accuracy


def make_jaccard_index_loss(smooth: float = 0.0000001) -> Callable:
    """
    Jaccard index training loss for segmentation like tasks.

    Default smoothness coefficient comes from Cloud-Net example.
    :param smooth: Small smoothing value to prevent zero division.
    :return: Callable Jaccard index loss function.
    """
    def jaccard_index_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        intersection = K.sum(y_true * y_pred, axis=(1, 2))
        y_true_sum = K.sum(y_true, axis=(1, 2))
        y_pred_sum = K.sum(y_pred, axis=(1, 2))

        jaccard = (intersection + smooth) / (
            y_true_sum + y_pred_sum - intersection + smooth
        )
        return 1 - jaccard

    return jaccard_index_loss


def make_jaccard_index_metric(smooth: float = 0.0000001) -> Callable:
    """
    Jaccard index metric for segmentation like tasks.

    Default smoothness coefficient comes from Cloud-Net example.
    Contrary to the make_jaccard_index_loss this operates on classes/categories
    not on probabilities.
    :param smooth: Small smoothing value to prevent zero division.
    :return: Callable Jaccard index metric function.
    """
    def jaccard_index_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        intersection = K.sum(
            K.round(K.clip(y_true * y_pred, 0, 1)), axis=(1, 2))
        y_true_sum = K.sum(K.round(y_true), axis=(1, 2))
        y_pred_sum = K.sum(K.round(y_pred), axis=(1, 2))

        return (intersection + smooth) / (
            y_true_sum + y_pred_sum - intersection + smooth
        )

    return jaccard_index_metric


def make_dice_coef_metric(smooth: float = 0.0000001) -> Callable:
    """
    Dice coefficient training loss for segmentation like tasks.

    Internally uses Jaccard index metric.
    :param smooth: Small smoothing value to prevent zero division.
    :return: Callable dice coef metric function.
    """
    jim = make_jaccard_index_metric(smooth)

    def dice_coeff_metric(y_true: np.ndarray, y_pred: np.ndarray):
        jim_score = jim(y_true, y_pred)
        return 2 * jim_score / (jim_score + 1)

    return dice_coeff_metric


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate recall score.
    :param y_true: True lables.
    :param y_pred: Predicted labels.
    :return: Recall score.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    ret = true_positives / (possible_positives + K.epsilon())
    return ret


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate precision score.
    :param y_true: True lables.
    :param y_pred: Predicted labels.
    :return: Precision score.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    ret = true_positives / (predicted_positives + K.epsilon())
    return ret


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate specificity score.
    :param y_true: True lables.
    :param y_pred: Predicted labels.
    :return: Specificity score.
    """
    y_true_neg = 1 - y_true
    y_pred_neg = 1 - y_pred
    true_negatives = K.sum(K.round(K.clip(y_true_neg * y_pred_neg, 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(y_true_neg, 0, 1)))
    ret = true_negatives / (possible_negatives + K.epsilon())
    return ret


# Same as make_dice_coef_metric() but calculated in a different way.
def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate f1 score.
    :param y_true: True lables.
    :param y_pred: Predicted labels.
    :return: F1 score.
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))
