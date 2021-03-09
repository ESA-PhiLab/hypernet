"""
Perform the inference of the unmixing model on the testing dataset.
In the case of unsupervised approach, the inference
is performed on the entire HSI.
"""

import os

import numpy as np
import tensorflow as tf
import yaml

from ml_intuition import enums
from ml_intuition.data import io, transforms
from ml_intuition.data.transforms import UNMIXING_TRANSFORMS
from ml_intuition.data.utils import get_central_pixel_spectrum
from ml_intuition.evaluation.performance_metrics import \
    calculate_unmixing_metrics, UNMIXING_TRAIN_METRICS
from ml_intuition.evaluation.time_metrics import timeit
from ml_intuition.models import Ensemble


def evaluate(data,
             model_path: str,
             dest_path: str,
             neighborhood_size: int,
             batch_size: int,
             endmembers_path: str,
             use_ensemble: bool = False,
             ensemble_copies: int = 1,
             voting: str = 'hard',
             noise_params: str = None,
             voting_model_params: str = None,
             voting_model: str = None,
             seed: int = 0):
    """
    Function for evaluating the trained model for the unmixing problem.

    :param model_path: Path to the model.
    :param data: Either path to the input data or the data dict.
    :param dest_path: Directory in which to store the calculated metrics
    :param neighborhood_size: Size of the spatial patch.
    :param batch_size: Size of the batch for inference.
    :param endmembers_path: Path to the endmembers file containing
        average reflectances for each class.
        Used only when use_unmixing is true.
    """
    model_name = os.path.basename(model_path)
    model = tf.keras.models.load_model(
        model_path, compile=True,
        custom_objects={metric.__name__: metric for metric in
                        UNMIXING_TRAIN_METRICS[model_name]})

    test_dict = data[enums.Dataset.TEST]

    min_, max_ = io.read_min_max(os.path.join(
        os.path.dirname(model_path), 'min-max.csv'))

    transformations = [transforms.MinMaxNormalize(min_=min_, max_=max_)]
    transformations += [t(**{'neighborhood_size': neighborhood_size}) for t
                        in UNMIXING_TRANSFORMS[model_name]]
    test_dict_transformed = transforms.apply_transformations(test_dict.copy(),
                                                             transformations)
    if 'dcae' in model_name:
        model.pop()

    if use_ensemble:
        model = Ensemble(model, voting=voting)
        noise_params = yaml.load(noise_params)
        model.generate_models_with_noise(copies=ensemble_copies,
                                         mean=noise_params['mean'],
                                         seed=seed)

        if voting == 'booster':
            train_dict_tr = data[enums.Dataset.TRAIN].copy()
            train_dict_tr = transforms.apply_transformations(train_dict_tr,
                                                             transformations)
            train_probabilities = model.predict_probabilities(
                train_dict_tr[enums.Dataset.DATA])
            model.train_ensemble_predictor(
                train_probabilities,
                data[enums.Dataset.TRAIN][enums.Dataset.LABELS],
                predictor=voting_model,
                model_params=voting_model_params)

    predict = timeit(model.predict)
    y_pred, inference_time = predict(
        test_dict_transformed[enums.Dataset.DATA],
        batch_size=batch_size)

    model_metrics = calculate_unmixing_metrics(**{
        'endmembers': np.load(endmembers_path)
        if endmembers_path is not None else None,
        'y_pred': y_pred,
        'y_true': test_dict[enums.Dataset.LABELS],
        'x_true': get_central_pixel_spectrum(
            test_dict_transformed[enums.Dataset.DATA],
            neighborhood_size)
    })

    model_metrics['inference_time'] = [inference_time]
    io.save_metrics(dest_path=dest_path,
                    file_name=enums.Experiment.INFERENCE_METRICS,
                    metrics=model_metrics)
