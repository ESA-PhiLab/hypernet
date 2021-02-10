"""
Perform the inference of the model on the dataset and return predictions.
"""

import os

import clize
import tensorflow as tf

from ml_intuition import enums
from ml_intuition.data import io, transforms
from ml_intuition.data.transforms import UNMIXING_TRANSFORMS
from ml_intuition.evaluation.performance_metrics import UNMIXING_TRAIN_METRICS


def predict(*,
            data,
            model_path: str,
            batch_size: int = 1024,
            dataset_to_predict: str = 'test',
            use_unmixing: bool = False,
            neighborhood_size: int = None):
    """
    Function for evaluating the trained model.

    :param data: Either path to the input data or the data dict.
    :param model_path: Path to the model.
    :param batch_size: Size of the batch for inference
    :param dataset_to_predict: Name of the dataset to predict,
        'train' or 'test'.
    :param use_unmixing: Boolean indicating whether
        to use unmixing functionality.
    :param neighborhood_size: Size of the local spatial extend of each sample.
    """
    if type(data) is str:
        set_dict = io.extract_set(data, dataset_to_predict)
    else:
        set_dict = data[dataset_to_predict]
    min_max_path = os.path.join(os.path.dirname(model_path), "min-max.csv")
    if os.path.exists(min_max_path):
        min_value, max_value = io.read_min_max(min_max_path)
    else:
        min_value, max_value = data[enums.DataStats.MIN], \
                               data[enums.DataStats.MAX]
    transformations = [transforms.MinMaxNormalize(min_=min_value,
                                                  max_=max_value)]

    custom_objects = None
    model_name = os.path.basename(model_path)

    if use_unmixing:
        transformations += [t(**{'neighborhood_size': neighborhood_size}) for t
                            in UNMIXING_TRANSFORMS[model_name]]
        custom_objects = {metric.__name__: metric for metric in
                          UNMIXING_TRAIN_METRICS[model_name]}

    if '2d' in os.path.basename(model_path) or 'deep' in os.path.basename(
            model_path):
        transformations.append(transforms.SpectralTransform())

    tf_set_dict = transforms.apply_transformations(set_dict.copy(),
                                                   transformations)

    model = tf.keras.models.load_model(model_path,
                                       compile=True,
                                       custom_objects=custom_objects)
    if 'dcae' in model_name:
        model.pop()

    y_pred = model.predict(tf_set_dict[enums.Dataset.DATA],
                           batch_size=batch_size)
    return y_pred


if __name__ == '__main__':
    clize.run(predict)
