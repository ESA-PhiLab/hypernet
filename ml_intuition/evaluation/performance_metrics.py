"""
All metrics that are calculated on the model's output.
"""
import os
from typing import Dict, List

import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from ml_intuition.data import utils


def mean_per_class_accuracy(y_true: np.ndarray,
                            y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate mean per class accuracy based on the confusion matrix.

    :param y_true: Labels as a one-dimensional numpy array.
    :param y_pred: Model's predictions as a one-dimensional numpy array.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    return conf_matrix.diagonal() / conf_matrix.sum(axis=1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metrics: list) -> \
        Dict[str, List[float]]:
    """
    Compute all metrics on predicted labels and labels.

    :param y_true: Labels as a one-dimensional numpy array.
    :param y_pred: Model's predictions as a one-dimensional numpy array.
    :param metrics: List of metrics functions.
    """
    return {metric_function.__name__:
                [metric_function(y_true, y_pred)] for metric_function in
            metrics}


CUSTOM_METRICS = [
    metrics.accuracy_score,
    metrics.balanced_accuracy_score,
    metrics.cohen_kappa_score,
    mean_per_class_accuracy,
]


def get_model_metrics(y_true, y_pred, inference_time):
    model_metrics = compute_metrics(y_true=y_true,
                                    y_pred=y_pred,
                                    metrics=CUSTOM_METRICS)
    model_metrics['inference_time'] = [inference_time]
    per_class_acc = {'Class_' + str(i):
                         [item] for i, item in enumerate(
        *model_metrics[mean_per_class_accuracy.__name__])}
    model_metrics.update(per_class_acc)
    model_metrics = utils.restructure_per_class_accuracy(model_metrics)
    return model_metrics




def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)