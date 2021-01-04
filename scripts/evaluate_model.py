"""
Perform the inference of the model on the testing dataset.
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


def evaluate(*,
             data,
             model_path: str,
             dest_path: str,
             n_classes: int,
             batch_size: int = 1024,
             use_ensemble: bool = False,
             ensemble_copies: int,
             voting: str = 'hard',
             noise: ('post', multi(min=0)),
             noise_sets: ('spost', multi(min=0)),
             noise_params: str = None,
             seed: int = 0):
    """
    Function for evaluating the trained model.

    :param model_path: Path to the model.
    :param data: Either path to the input data or the data dict.
    :param dest_path: Directory in which to store the calculated metrics
    :param n_classes: Number of classes.
    :param batch_size: Size of the batch for inference
    :param use_ensemble: Use ensemble for prediction.
    :param ensemble_copies: Number of model copies for the ensemble.
    :param voting: Method of ensemble voting. If ‘hard’, uses predicted class
            labels for majority rule voting. Else if ‘soft’, predicts the class
            label based on the argmax of the sums of the predicted probabilities.
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
    :param seed: Seed for RNG
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
                       transforms.MinMaxNormalize(min_=min_value, max_=max_value)]

    if '2d' in os.path.basename(model_path) or 'deep' in os.path.basename(model_path):
        transformations.append(transforms.SpectralTransform())

    transformations = transformations + get_noise_functions(noise, noise_params) \
        if enums.Dataset.TEST in noise_sets else transformations

    test_dict = transforms.apply_transformations(test_dict, transformations)

    model = tf.keras.models.load_model(model_path, compile=True)
    if use_ensemble:
        model = Ensemble(model, voting=voting)

        if ensemble_copies is not None:
            noise_params = yaml.load(noise_params)
            model.generate_models_with_noise(copies=ensemble_copies,
                                             mean=noise_params['mean'],
                                             seed=seed)
        if voting == 'classifier':
            if type(data) is str:
                train_dict = io.extract_set(data, enums.Dataset.TRAIN)
            else:
                train_dict = data[enums.Dataset.TRAIN]
            train_dict = transforms.apply_transformations(train_dict, transformations)
            train_probabilities = model.predict_probabilities(train_dict[enums.Dataset.DATA])
            model.train_ensemble_predictor(train_probabilities, train_dict[enums.Dataset.LABELS])

    predict = timeit(model.predict)
    y_pred, inference_time = predict(test_dict[enums.Dataset.DATA],
                                     batch_size=batch_size)

    if voting == 'classifier':
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
