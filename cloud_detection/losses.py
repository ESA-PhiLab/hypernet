"""
Custom loss functions and metrics for segmentation.

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

import tensorflow.keras.backend as K
import numpy as np

# Import all other metrics used by the model
# to enable testing it.
# noqa # pylint: disable=unused-import
from tensorflow.keras.metrics import binary_crossentropy, binary_accuracy


class JaccardIndexLoss:
    """Jaccard index loss for segmentation like tasks."""

    def __init__(self, smooth: float = 0.0000001):
        """
        Create Jaccard index loss callable class.
        Default smoothness coefficient comes from Cloud-Net example.

        :param smooth: Small smoothing value to prevent zero division.
        """
        self.__name__ = "jaccard_index_loss"
        self._smooth: float = smooth

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Jaccard index loss.

        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :return: Jaccard index loss score value
        """
        intersection = K.sum(y_true * y_pred, axis=(1, 2))
        y_true_sum = K.sum(y_true, axis=(1, 2))
        y_pred_sum = K.sum(y_pred, axis=(1, 2))

        jaccard = (intersection + self._smooth) / (
            y_true_sum + y_pred_sum - intersection + self._smooth
        )
        return 1 - jaccard


class JaccardIndexMetric:
    """
    Jaccard index metric for segmentation like tasks.
    Contrary to the JaccardIndexLoss this operates on classes/categories
    not on probabilities.
    """

    def __init__(self, smooth: float = 0.0000001):
        """
        Create Jaccard index metric loss callable class.
        Default smoothness coefficient comes from Cloud-Net example.

        :param smooth: Small smoothing value to prevent zero division.
        """
        self.__name__ = "jaccard_index_metric"
        self._smooth: float = smooth

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Jaccard index metric.

        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :return: Jaccard index metric score value.
        """
        intersection = K.sum(
            K.round(K.clip(y_true * y_pred, 0, 1)), axis=(1, 2))
        y_true_sum = K.sum(K.round(y_true), axis=(1, 2))
        y_pred_sum = K.sum(K.round(y_pred), axis=(1, 2))

        return (intersection + self._smooth) / (
            y_true_sum + y_pred_sum - intersection + self._smooth
        )


class DiceCoefMetric:
    """
    Dice coefficient metric for segmentation like tasks.
    Internally uses Jaccard index metric.
    """

    def __init__(self, smooth: float = 0.0000001):
        """
        Create dice coef metric callable class.
        Default smoothness coefficient comes from Cloud-Net example.

        :param smooth: Small smoothing value to prevent zero division.
        """
        self.__name__ = "dice_coeff_metric"
        self._jim = JaccardIndexMetric(smooth)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calcualate dice coef metric.

        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :return: Dice coef metric score value.
        """
        jim_score = self._jim(y_true, y_pred)
        return 2 * jim_score / (jim_score + 1)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate recall score.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: Recall score.
    """
    true_positives = K.sum(
        K.round(K.clip(y_true * y_pred, 0, 1)), axis=(1, 2)
    )
    possible_positives = K.sum(
        K.round(K.clip(y_true, 0, 1)), axis=(1, 2)
    )
    ret = true_positives / (possible_positives + K.epsilon())
    return ret


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate precision score.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: Precision score.
    """
    true_positives = K.sum(
        K.round(K.clip(y_true * y_pred, 0, 1)), axis=(1, 2)
    )
    predicted_positives = K.sum(
        K.round(K.clip(y_pred, 0, 1)), axis=(1, 2)
    )
    ret = true_positives / (predicted_positives + K.epsilon())
    return ret


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate specificity score.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: Specificity score.
    """
    y_true_neg = 1 - y_true
    y_pred_neg = 1 - y_pred
    true_negatives = K.sum(
        K.round(K.clip(y_true_neg * y_pred_neg, 0, 1)), axis=(1, 2)
    )
    possible_negatives = K.sum(
        K.round(K.clip(y_true_neg, 0, 1)), axis=(1, 2)
    )
    ret = true_negatives / (possible_negatives + K.epsilon())
    return ret


# Same as DiceCoefMetric() but calculated in a different way.
def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate f1 score.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: F1 score.
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))
