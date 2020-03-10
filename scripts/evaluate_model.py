"""
Perform the inference of the model on the testing dataset.
"""

import json
import os

import clize
import numpy as np
import tensorflow as tf
from sklearn import metrics

from ml_intuition import enums
from ml_intuition.data import io, transforms, utils
from ml_intuition.evaluation.performance_metrics import (
    compute_metrics, mean_per_class_accuracy)
from ml_intuition.evaluation.time_metrics import timeit


def evaluate(*,
             model_path: str,
             data_path: str,
             verbose: int,
             n_classes: int):
    """
    Function for evaluating the trained model.

    :param model_path: Path to the model.
    :param data_path: Path to the input data.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param n_classes: Number of classes.
    """
    test_dict = io.load_data(data_path, enums.Dataset.TEST)
    test_dataset, n_test =\
        utils.extract_dataset(1,
                              test_dict,
                              [transforms.SpectralTranform(n_classes),
                               transforms.MinMaxNormalize(min_=test_dict[enums.DataStats.MIN],
                                                          max_=test_dict[enums.DataStats.MAX])])

    model = tf.keras.models.load_model(model_path, compile=True)
    model.predict = timeit(model.predict)
    y_pred, inference_time = model.predict(x=test_dataset.make_one_shot_iterator(),
                                           verbose=verbose,
                                           steps=n_test // 1)

    y_pred = tf.Session().run(tf.argmax(y_pred, axis=-1))
    y_true = test_dict[enums.Dataset.LABELS]

    custom_metrics = [
        metrics.accuracy_score,
        metrics.balanced_accuracy_score,
        metrics.cohen_kappa_score,
        mean_per_class_accuracy,
        metrics.confusion_matrix
    ]
    model_metrics = compute_metrics(y_true=y_true,
                                    y_pred=y_pred,
                                    metrics=custom_metrics)
    model_metrics['inference_time'] = [inference_time]
    per_class_acc = {'Class_' + str(i):
                     [item] for i, item in enumerate(*model_metrics[mean_per_class_accuracy.__name__])}
    model_metrics.update(per_class_acc)

    np.savetxt(os.path.join(os.path.dirname(model_path),
                            metrics.confusion_matrix.__name__ + '.csv'),
               *model_metrics[metrics.confusion_matrix.__name__], delimiter=',', fmt='%d')

    del model_metrics[mean_per_class_accuracy.__name__]
    del model_metrics[metrics.confusion_matrix.__name__]

    io.save_metrics(dest_path=os.path.dirname(model_path),
                    metric_key='inference_metrics.csv',
                    metrics=model_metrics)


if __name__ == '__main__':
    clize.run(evaluate)
