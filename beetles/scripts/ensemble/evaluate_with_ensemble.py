"""
Evaluate a dataset using an ensemble.
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
             y_pred,
             data,
             dest_path: str,
             model_path: str,
             voting: str = 'hard'):
    """
    Function for evaluating the trained model.

    :param y_pred: Predictions of all the models to be provided into an
        ensemble to vote
    :param data: Either path to the input data or the data dict.
    :param dest_path: Directory in which to store the calculated metrics
    :param model_path: Path to the model.
    :param voting: Method of ensemble voting. If ‘hard’, uses predicted class
            labels for majority rule voting. Else if ‘soft’, predicts the class
            label based on the argmax of the sums of the predicted probabilities.
    """
    ensemble = Ensemble(voting=voting)
    vote = timeit(ensemble.vote)
    y_pred, voting_time = vote(y_pred)

    y_true = data[enums.Dataset.TEST][enums.Dataset.LABELS]
    y_true = np.argmax(y_true, axis=-1)
    model_metrics = get_model_metrics(y_true, y_pred)
    model_metrics['inference_time'] = [voting_time]
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
