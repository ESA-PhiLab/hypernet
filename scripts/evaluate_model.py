"""
Perform the inference of the model on the testing dataset.
"""

import os
from typing import Dict, Union, List

import clize
import numpy as np
import tensorflow as tf
from sklearn import metrics

from ml_intuition import enums
from ml_intuition.data import io, transforms, utils
from ml_intuition.evaluation.performance_metrics import (
    compute_metrics, mean_per_class_accuracy)
from ml_intuition.evaluation.time_metrics import timeit

BATCH_SIZE = 1


def evaluate(*,
             model_path: str,
             data: Union[str, Dict],
             dest_path: str,
             verbose: int = 1,
             n_classes: int,
             augmentation: List,
             params: Dict):
    """
    Function for evaluating the trained model.

    :param model_path: Path to the model.
    :param data: Either path to the input data or the data dict.
    :param dest_path: Directory in which to store the calculated matrics
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param n_classes: Number of classes.
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
    test_dataset, n_test = \
        utils.create_tf_dataset(BATCH_SIZE,
                                test_dict,
                                [transforms.SpectralTransform(),
                                 transforms.OneHotEncode(n_classes=n_classes),
                                 transforms.MinMaxNormalize(
                                     min_=min_value,
                                     max_=max_value)])

    if augmentation is not None:
        params = json.loads(*params)
        for fun_aug in get_augmentation(augmentation):
            fun_aug(data_source[Dataset.TEST][Dataset.DATA], params)

    model = tf.keras.models.load_model(model_path, compile=True)
    model.predict = timeit(model.predict)
    y_pred, inference_time = model.predict(
        x=test_dataset.make_one_shot_iterator(),
        verbose=verbose,
        steps=n_test // BATCH_SIZE)

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
                     [item] for i, item in enumerate(
        *model_metrics[mean_per_class_accuracy.__name__])}
    model_metrics.update(per_class_acc)

    np.savetxt(os.path.join(os.path.dirname(model_path),
                            metrics.confusion_matrix.__name__ + '.csv'),
               *model_metrics[metrics.confusion_matrix.__name__], delimiter=',',
               fmt='%d')

    del model_metrics[metrics.confusion_matrix.__name__]
    model_metrics = utils.restructure_per_class_accuracy(model_metrics)

    io.save_metrics(dest_path=dest_path,
                    file_name=enums.Experiment.INFERENCE_METRICS,
                    metrics=model_metrics)


if __name__ == '__main__':
    clize.run(evaluate)
