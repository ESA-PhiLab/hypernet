"""
Perform the inference of the model on the testing dataset.
"""

import json
import os

import clize
import tensorflow as tf
from scripts import metrics

from ml_intuition.data import io, transforms, utils


def evaluate(*,
             model_path: str,
             data_path: str,
             verbose: int,
             sample_size: int,
             n_classes: int):
    """
    Function for evaluating the trained model.

    :param model_path: Path to the model.
    :param data_path: Path to the input data.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param sample_size: Size of the input sample.
    :param n_classes: Number of classes.
    """
    test_dict = io.load_data(data_path, utils.Dataset.TEST)
    test_dataset, n_test =\
        utils.extract_dataset(1,
                              test_dict,
                              [transforms.SpectralTranform(sample_size,
                                                           n_classes)])

    model = tf.keras.models.load_model(model_path, compile=True)
    model.predict = metrics.Metrics.timeit(model.predict)
    y_pred, inference_time = model.predict(x=test_dataset.make_one_shot_iterator(),
                                           verbose=verbose,
                                           steps=n_test // 1)

    y_pred = tf.Session().run(tf.argmax(y_pred, axis=-1))
    y_true = test_dict[utils.Dataset.LABELS]
    metrics.Metrics().compute_metrics(y_true=y_true, y_pred=y_pred,
                                      inference_time=inference_time)\
        .save_metrics(dest_path=os.path.dirname(model_path))


if __name__ == '__main__':
    clize.run(evaluate)
