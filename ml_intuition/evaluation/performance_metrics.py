"""
All metrics that are calculated on the model's output.
"""
from typing import Dict, List, Union

import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from ml_intuition.data import utils


def convert_to_tensor(metric_function):
    def wrapper(y_true: np.ndarray, y_pred: np.ndarray):
        if not isinstance(y_true, tf.Tensor):
            y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
        if not isinstance(y_pred, tf.Tensor):
            y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
        with tf.Session() as session:
            return metric_function(y_true=y_true, y_pred=y_pred).eval(session=session)

    return wrapper


def spectral_information_divergence_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the spectral information divergence loss, which is based on the divergence in information theory.
    :param y_true: Labels as two dimensional abundances or original input array of shape:
    [n_samples, n_classes], [n_samples, n_bands].
    :param y_pred: Predicted abundances or reconstructed input array of shape:
    [n_samples, n_classes], [n_samples, n_bands].
    :return: The spectral information divergence loss.
    """
    y_true_row_sum = tf.reduce_sum(y_true, 1)
    y_pred_row_sum = tf.reduce_sum(y_pred, 1)
    y_true = y_true / tf.reshape(y_true_row_sum, (-1, 1))
    y_pred = y_pred / tf.reshape(y_pred_row_sum, (-1, 1))
    y_true, y_pred = tf.keras.backend.clip(y_true, tf.keras.backend.epsilon(), 1), \
                     tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1)
    loss = tf.reduce_sum(y_true * tf.log(y_true / y_pred)) + tf.reduce_sum(y_pred * tf.log(y_pred / y_true))
    return loss


def sum_per_class_rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the sum of per class root-mean square error.
    :param y_true: Labels as two dimensional abundances array of shape: [n_samples, n_classes].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes].
    :return: The sum of per class root-mean square error.
    """
    return tf.reduce_sum(per_class_rmse(y_true=y_true, y_pred=y_pred))


def per_class_rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the per class root-mean square error vector.
    :param y_true: Labels as two dimensional abundances array of shape: [n_samples, n_classes].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes].
    :return: The root-mean square error vector.
    """
    return tf.sqrt(tf.reduce_mean((y_true - y_pred) ** 2, 0))


def average_angle_spectral_mapper(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the average angle spectral mapper value.
    :param y_true: Labels as two dimensional abundances array of shape: [n_samples, n_classes]
        or original input pixel and its reconstruction of shape: [n_samples, n_bands].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes]
        or original input pixel and its reconstruction of shape: [n_samples, n_bands].
    :return: The root-mean square abundance angle distance error.
    """
    numerator = tf.reduce_sum(tf.multiply(y_true, y_pred), 1)
    y_true_len = tf.sqrt(tf.reduce_sum(tf.square(y_true), 1))
    y_pred_len = tf.sqrt(tf.reduce_sum(tf.square(y_pred), 1))
    denominator = tf.multiply(y_true_len, y_pred_len)
    loss = tf.reduce_mean(tf.acos(tf.clip_by_value(numerator / denominator, -1, 1)))
    return loss


def overall_rms_abundance_angle_distance(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the root-mean square abundance angle distance, which measures the similarity
    between the original abundance fractions and the predicted ones.
    It utilizes the inverse of cosine function at the range [0, pi], which means that
    the domain of arccos is in the range [-1; 1], that is why the "tf.clip_by_value" method is used.
    For the identical abundances the numerator / denominator is 1 and arccos(1) is 0, which resembles the perfect score.
    :param y_true: Labels as two dimensional abundances array of shape: [n_samples, n_classes].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes].
    :return: The root-mean square abundance angle distance error.
    """
    numerator = tf.reduce_sum(tf.multiply(y_true, y_pred), 1)
    y_true_len = tf.sqrt(tf.reduce_sum(tf.square(y_true), 1))
    y_pred_len = tf.sqrt(tf.reduce_sum(tf.square(y_pred), 1))
    denominator = tf.multiply(y_true_len, y_pred_len)
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.acos(tf.clip_by_value(numerator / denominator, -1, 1)))))
    return loss


def overall_rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the root-mean square error, which measures the similarity
    between the original abundance fractions and the predicted ones.
    :param y_true: Labels as two dimensional abundances array of shape: [n_samples, n_classes].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes].
    :return: The root-mean square error.
    """
    return tf.sqrt(tf.reduce_mean((y_true - y_pred) ** 2))


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


DEFAULT_METRICS = [
    metrics.accuracy_score,
    metrics.balanced_accuracy_score,
    metrics.cohen_kappa_score,
    mean_per_class_accuracy,
]

DEFAULT_FAIR_METRICS = [
    metrics.accuracy_score,
    metrics.balanced_accuracy_score,
    metrics.cohen_kappa_score
]

UNMIXING_METRICS = {
    'TRAIN': {'dcae': {spectral_information_divergence_loss.__name__: spectral_information_divergence_loss,
                       overall_rmse.__name__: overall_rmse},
              'cnn': {overall_rmse.__name__: overall_rmse,
                      overall_rms_abundance_angle_distance.__name__: overall_rms_abundance_angle_distance,
                      sum_per_class_rmse.__name__: sum_per_class_rmse}},
    'TEST': {'dcae': {'aRMSE': overall_rmse,
                      'aSAM': average_angle_spectral_mapper},
             'cnn': {'overallRMSE': overall_rmse,
                     'rmsAAD': overall_rms_abundance_angle_distance,
                     'sumRMSE': sum_per_class_rmse}}
}
UNMIXING_LOSSES = {
    'dcae': spectral_information_divergence_loss,
    'cnn': 'mse'
}


def get_loss(model_name: str, use_unmixing: bool = True, **kwargs) -> Union[str, object]:
    loss = 'categorical_crossentropy'
    if use_unmixing:
        loss = UNMIXING_LOSSES[model_name.split('_')[-1]]
    return loss


def get_unmixing_metrics(model_name: str, use_unmixing: bool = False, mode: str = 'TRAIN') -> Dict[str, object]:
    unmixing_metrics = {}
    if use_unmixing:
        try:
            unmixing_metrics = UNMIXING_METRICS[mode][model_name.split('_')[-1]]
        except KeyError:
            print(f'The model name: {model_name} or mode: {mode} for unmixing is incorrect.')
    return unmixing_metrics


def calculate_unmixing_metrics(model_name: str, **kwargs) -> Dict[str, List[float]]:
    """
    Calculate the metrics for unmixing problem.

    :param model_name: Name of the unmixing model.
    :param kwargs: Additional keyword arguments.
    """
    model_metrics = {}
    print(kwargs['y_pred'].shape)
    for f_name, f_metric in get_unmixing_metrics(model_name, True, 'TEST').items():
        model_metrics[f_name] = [float(convert_to_tensor(f_metric)
                                       (y_true=kwargs['y_true'], y_pred=kwargs['y_pred']))]
    for class_idx, class_rmse in enumerate(convert_to_tensor(per_class_rmse)(
            y_true=kwargs['y_true'], y_pred=kwargs['y_pred'])):
        model_metrics[f'Class_{class_idx}_rmse'] = [float(class_rmse)]
    if kwargs['endmembers'] is not None:
        # Calculate the reconstruction RMSE and SID losses:
        x_pred = np.matmul(kwargs['y_pred'], kwargs['endmembers'].T)
        model_metrics['rRMSE'] = [float(convert_to_tensor(overall_rmse)
                                        (y_true=kwargs['x_true'], y_pred=x_pred))]
        model_metrics['rSID'] = [float(convert_to_tensor(spectral_information_divergence_loss)
                                       (y_true=kwargs['x_true'], y_pred=x_pred))]
    return model_metrics


def get_model_metrics(y_true, y_pred, metrics_to_compute: List = None) \
        -> Dict[str, List[float]]:
    """
    Calculate provided metrics and store them in a Dict
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param inference_time: Prediction time, defaults to None
    :param metrics_to_compute: Metrics which will be computed, defaults to None
    :return: Dictionary with metric name as key and metric value as value
    """
    metrics_to_compute = DEFAULT_METRICS if metrics_to_compute is None else \
        metrics_to_compute
    model_metrics = compute_metrics(y_true=y_true,
                                    y_pred=y_pred,
                                    metrics=metrics_to_compute)
    if mean_per_class_accuracy.__name__ in model_metrics:
        per_class_acc = {'Class_' + str(i):
                             [item] for i, item in enumerate(
            *model_metrics[mean_per_class_accuracy.__name__])}
        model_metrics.update(per_class_acc)
        model_metrics = utils.restructure_per_class_accuracy(model_metrics)
    return model_metrics


def get_fair_model_metrics(conf_matrix, labels_in_train) -> Dict[str, List[float]]:
    """
    Recalculate model metrics discarding classes which where not present in the
    training set
    :param conf_matrix: Original confusion matrix containing all classes
    :param labels_in_train: Labels which were present in the training set
    :return: Recalculated metrics as Dict
    """
    conf_matrix = conf_matrix[labels_in_train, :]
    conf_matrix = conf_matrix[:, labels_in_train]
    all_preds = []
    all_targets = []
    for row_id in range(conf_matrix.shape[0]):
        preds = []
        targets = []
        for column_id in range(conf_matrix.shape[1]):
            for i in range(int(conf_matrix[row_id, column_id])):
                preds.append(column_id)
            targets = [row_id for _ in range(len(preds))]
        all_preds.append(preds)
        all_targets.append(targets)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    return get_model_metrics(y_true, y_pred,
                             metrics_to_compute=DEFAULT_FAIR_METRICS)
