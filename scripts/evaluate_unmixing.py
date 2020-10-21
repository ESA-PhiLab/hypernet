"""
Perform the inference of the unmixing model on the testing dataset.
In the case of unsupervised approach, the inference
is performed on the entire HSI.
"""

import os

import numpy as np
import tensorflow as tf

from ml_intuition import enums
from ml_intuition.data import io, transforms
from ml_intuition.data.utils import get_central_pixel_spectrum
from ml_intuition.evaluation.performance_metrics import \
    calculate_unmixing_metrics, cnn_rmse, \
    overall_rms_abundance_angle_distance, sum_per_class_rmse, \
    spectral_information_divergence_loss
from ml_intuition.evaluation.time_metrics import timeit
from ml_intuition.models import unmixing_pixel_based_dcae, \
    unmixing_cube_based_dcae, unmixing_cube_based_cnn, \
    unmixing_pixel_based_cnn, unmixing_rnn_supervised

SUPERVISED_TRANSFORMS = {
    unmixing_pixel_based_cnn.__name__: transforms.SpectralTransform,
    unmixing_cube_based_cnn.__name__: transforms.SpectralTransform,
    unmixing_rnn_supervised.__name__: transforms.RNNSpectralInputTransform
}


def evaluate_dcae(**kwargs):
    """
    Evaluate the deep convolutional autoencoder (DCAE).

    :param kwargs: The keyword arguments containing specific
        hyperparameters and data.
    """
    model = tf.keras.models.load_model(
        kwargs['model_path'], compile=True,
        custom_objects={
            spectral_information_divergence_loss.__name__:
                spectral_information_divergence_loss})

    model.pop()  # Drop the decoder of the already trained autoencoder.
    predict = timeit(model.predict)
    y_pred, inference_time = predict(kwargs['data']['data'],
                                     batch_size=kwargs['batch_size'])

    model_metrics = calculate_unmixing_metrics(**{
        'endmembers': np.load(kwargs['endmembers_path']),
        'y_pred': y_pred, 'y_true': kwargs['data']['labels'],
        'x_true': get_central_pixel_spectrum(kwargs['data']['data'],
                                             kwargs['neighborhood_size'])
    })
    model_metrics['inference_time'] = [inference_time]
    io.save_metrics(dest_path=kwargs['dest_path'],
                    file_name=enums.Experiment.INFERENCE_METRICS,
                    metrics=model_metrics)


def evaluate_supervised(**kwargs):
    """
    Evaluate the supervised models.

    :param kwargs: The keyword arguments containing specific
        hyperparameters and data.
    """
    model = tf.keras.models.load_model(
        kwargs['model_path'], compile=True,
        custom_objects={cnn_rmse.__name__: cnn_rmse,
                        overall_rms_abundance_angle_distance.__name__:
                            overall_rms_abundance_angle_distance,
                        sum_per_class_rmse.__name__: sum_per_class_rmse})

    test_dict = kwargs['data'][enums.Dataset.TEST]

    min_value, max_value = io.read_min_max(os.path.join(
        os.path.dirname(kwargs['model_path']), 'min-max.csv'))

    test_dict = transforms.apply_transformations(
        test_dict, [SUPERVISED_TRANSFORMS[kwargs['model_name']](),
                    transforms.MinMaxNormalize(min_=min_value,
                                               max_=max_value)])
    predict = timeit(model.predict)
    y_pred, inference_time = predict(test_dict[enums.Dataset.DATA],
                                     batch_size=kwargs['batch_size'])

    model_metrics = calculate_unmixing_metrics(**{
        'endmembers': None,
        'y_pred': y_pred, 'y_true': test_dict[enums.Dataset.LABELS]})

    model_metrics['inference_time'] = [inference_time]
    io.save_metrics(dest_path=kwargs['dest_path'],
                    file_name=enums.Experiment.INFERENCE_METRICS,
                    metrics=model_metrics)


EVALUATE_FUNCTION = {
    unmixing_cube_based_dcae.__name__: evaluate_dcae,
    unmixing_pixel_based_dcae.__name__: evaluate_dcae,
    unmixing_cube_based_cnn.__name__: evaluate_supervised,
    unmixing_pixel_based_cnn.__name__: evaluate_supervised,
    unmixing_rnn_supervised.__name__: evaluate_supervised
}


def evaluate(data,
             model_path: str,
             dest_path: str,
             n_classes: int,
             neighborhood_size: int = None,
             batch_size: int = 1024,
             endmembers_path: str = None):
    """
    Function for evaluating the trained model for the unmixing problem.

    :param model_path: Path to the model.
    :param data: Either path to the input data or the data dict.
    :param dest_path: Directory in which to store the calculated metrics
    :param n_classes: Number of classes.
    :param neighborhood_size: Size of the spatial patch.
    :param batch_size: Size of the batch for inference.
    :param endmembers_path: Path to the endmembers file containing
        average reflectances for each class.
        Used only when use_unmixing is true.
    """
    model_name = os.path.basename(model_path)
    EVALUATE_FUNCTION[model_name](**{
        'data': data, 'model_path': model_path, 'dest_path': dest_path,
        'n_classes': n_classes, 'neighborhood_size': neighborhood_size,
        'batch_size': batch_size, 'endmembers_path': endmembers_path,
        'model_name': model_name
    })
