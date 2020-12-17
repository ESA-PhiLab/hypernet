"""
Perform the inference of the model on the provided dataset.
"""

import os

import clize
import yaml
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
from ml_intuition.models import Ensemble


def predict(*,
            data,
            model_path: str,
            n_classes: int,
            batch_size: int = 1024,
            noise: ('post', multi(min=0)),
            noise_sets: ('spost', multi(min=0)),
            noise_params: str = None):
    """
    Function for evaluating the trained model.

    :param data: Either path to the input data or the data dict.
    :param model_path: Path to the model.
    :param n_classes: Number of classes.
    :param batch_size: Size of the batch for inference.
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
    transformations = [transforms.OneHotEncode(n_classes=n_classes),
                       transforms.MinMaxNormalize(min_=min_value,
                                                  max_=max_value)]

    if '2d' in os.path.basename(model_path) or 'deep' in os.path.basename(
            model_path):
        transformations.append(transforms.SpectralTransform())

    transformations = transformations + get_noise_functions(noise, noise_params) \
        if enums.Dataset.TEST in noise_sets else transformations

    test_dict = transforms.apply_transformations(test_dict, transformations)

    model = tf.keras.models.load_model(model_path, compile=True)

    y_pred = model.predict(test_dict[enums.Dataset.DATA],
                           batch_size=batch_size)

    return y_pred


if __name__ == '__main__':
    clize.run(predict)
