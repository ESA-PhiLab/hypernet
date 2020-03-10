"""
All metrics that are calculated on the model's output.
"""

import numpy as np
from sklearn.metrics import confusion_matrix


def mean_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate mean per class accuracy based on the confusion matrix.

    :param y_true: Labels as a one-dimensional numpy array.
    :param y_pred: Model's predictions as a one-dimensional numpy array.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    return conf_matrix.diagonal() / conf_matrix.sum(axis=1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metrics: list) -> dict:
    """
    Compute all metrics on predicted labels and labels.

    :param y_true: Labels as a one-dimensional numpy array.
    :param y_pred: Model's predictions as a one-dimensional numpy array.
    """
    return {metric_function.__name__:
            [metric_function(y_true, y_pred)] for metric_function in metrics}
