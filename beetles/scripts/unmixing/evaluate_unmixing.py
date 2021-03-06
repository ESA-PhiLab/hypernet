"""
Perform the inference of the unmixing models on the test dataset.
"""

import os

import numpy as np
import tensorflow as tf

from ml_intuition import enums
from ml_intuition.data import io, transforms
from ml_intuition.data.transforms import UNMIXING_TRANSFORMS
from ml_intuition.data.utils import get_central_pixel_spectrum
from ml_intuition.evaluation.performance_metrics import \
    calculate_unmixing_metrics, UNMIXING_TRAIN_METRICS
from ml_intuition.evaluation.time_metrics import timeit


def evaluate(data,
             model_path: str,
             dest_path: str,
             neighborhood_size: int,
             batch_size: int,
             endmembers_path: str):
    """
    Function for evaluating the trained model for the unmixing problem.

    :param model_path: Path to the model.
    :param data: The data dictionary containing the subset for testing.
    :param dest_path: Directory in which to store the calculated metrics.
    :param neighborhood_size: Size of the spatial patch.
    :param batch_size: Size of the batch for inference.
    :param endmembers_path: Path to the endmembers matrix file,
        containing the average reflectances for each endmember,
        i.e., the pure spectra.
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
