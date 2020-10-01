"""
Perform the inference of the model on the testing dataset.
"""

import os

import clize
import numpy as np
import tensorflow as tf
from clize.parameters import multi
from sklearn.metrics import confusion_matrix

from ml_intuition import enums
from ml_intuition.data import io, transforms
from ml_intuition.data.noise import get_noise_functions
from ml_intuition.evaluation.performance_metrics import get_model_metrics, \
    get_fair_model_metrics
from ml_intuition.evaluation.time_metrics import timeit


def evaluate(*,
             data,
             model_path: str,
             dest_path: str,
             n_classes: int,
             batch_size: int = 1024,
             noise: ('post', multi(min=0)),
             noise_sets: ('spost', multi(min=0)),
             noise_params: str = None):
    """
    Function for evaluating the trained model.

    :param model_path: Path to the model.
    :param data: Either path to the input data or the data dict.
    :param dest_path: Directory in which to store the calculated metrics
    :param n_classes: Number of classes.
    :param batch_size: Size of the batch for inference
    :param noise: List containing names of used noise injection methods
        that are performed after the normalization transformations.
    :param noise_sets: List of sets that are affected by the noise injection.
        For this module single element can be "test".
    :param noise_params: JSON containing the parameters
        setting of noise injection methods.
        Exemplary value for this parameter: "{"mean": 0, "std": 1, "pa": 0.1}".
        This JSON should include all parameters for noise injection
        functions that are specified in the noise argument.
        For the accurate description of each parameter, please
        refer to the ml_intuition/data/noise.py module.
    """
    if type(data) is str:
        test_dict = io.extract_set(data, enums.Dataset.TEST)
    else:
        test_dict = data[enums.Dataset.TEST]
    min_max_path = os.path.join(os.path.dirname(model_path), "min-max.csv")
    if os.path.exists(min_max_path):
        min_value, max_value = io.read_min_max(min_max_path)
    else:
        min_value, max_value = data[enums.DataStats.MIN], \
                               data[enums.DataStats.MAX]

    transformations = [transforms.SpectralTransform(),
                       transforms.OneHotEncode(n_classes=n_classes),
                       transforms.MinMaxNormalize(min_=min_value, max_=max_value)]
    transformations = transformations + get_noise_functions(noise, noise_params) \
        if enums.Dataset.TEST in noise_sets else transformations

    test_dict = transforms.apply_transformations(test_dict, transformations)

    model = tf.keras.models.load_model(model_path, compile=True)

    predict = timeit(model.predict)
    y_pred, inference_time = predict(test_dict[enums.Dataset.DATA],
                                     batch_size=batch_size)

    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(test_dict[enums.Dataset.LABELS], axis=-1)

    model_metrics = get_model_metrics(y_true, y_pred)
    model_metrics['inference_time'] = [inference_time]
    conf_matrix = confusion_matrix(y_true, y_pred)
    io.save_metrics(dest_path=dest_path,
                    file_name=enums.Experiment.INFERENCE_METRICS,
                    metrics=model_metrics)
    io.save_confusion_matrix(conf_matrix, dest_path)
    if enums.Splits.GRIDS in model_path:
        if type(data) is str:
            train_dict = io.extract_set(data, enums.Dataset.TRAIN)
            labels_in_train = np.unique(train_dict[enums.Dataset.LABELS])
        else:
            train_labels = data[enums.Dataset.TRAIN][enums.Dataset.LABELS]
            if train_labels.ndim > 1:
                train_labels = np.argmax(train_labels, axis=-1)
            labels_in_train = np.unique(train_labels)
        fair_metrics = get_fair_model_metrics(conf_matrix, labels_in_train)
        io.save_metrics(dest_path=dest_path,
                        file_name=enums.Experiment.INFERENCE_FAIR_METRICS,
                        metrics=fair_metrics)


if __name__ == '__main__':
    clize.run(evaluate)
