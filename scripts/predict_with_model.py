"""
Perform the inference of the model on the testing dataset.
"""

import os

import clize
import tensorflow as tf

from ml_intuition import enums
from ml_intuition.data import io, transforms


def predict(*,
            data,
            model_path: str,
            batch_size: int = 1024,
            dataset_to_predict: str = 'test'):
    """
    Function for evaluating the trained model.

    :param model_path: Path to the model.
    :param data: Either path to the input data or the data dict.
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

    if '2d' in os.path.basename(model_path) or 'deep' in os.path.basename(
            model_path):
        transformations.append(transforms.SpectralTransform())

    set_dict = transforms.apply_transformations(set_dict, transformations)

    model = tf.keras.models.load_model(model_path, compile=True)
    print('before pred:', set_dict[enums.Dataset.DATA].shape)
    y_pred = model.predict(set_dict[enums.Dataset.DATA],
                           batch_size=batch_size)
    print('after pred', y_pred.shape)
    # y_pred = np.argmax(y_pred, axis=-1)

    return y_pred


if __name__ == '__main__':
    clize.run(predict)
