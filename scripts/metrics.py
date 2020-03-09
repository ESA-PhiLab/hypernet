"""
All metrics that are calculated on the model's output.
"""
import csv
import os
import time
from time import time
from typing import Dict, Union

import numpy as np
from sklearn import metrics
from tensorflow.keras.callbacks import Callback


class TimeHistory(Callback):
    """
    Custom keras callback logging duration of each epoch.
    """

    def on_train_begin(self, logs: dict = {}):
        self.on_train_begin_time = time()
        self.times = []
        self.average = []

    def on_epoch_begin(self, batch: int, logs: dict = {}):
        self.epoch_time_start = time()

    def on_epoch_end(self, batch: int, logs: dict = {}):
        self.times.append(time() - self.on_train_begin_time)
        self.average.append(time() - self.epoch_time_start)


class Metrics:
    """
    Compute metrcis given model predictions and target labels.
    """
    METRICS = [
        metrics.accuracy_score,
        metrics.balanced_accuracy_score,
        metrics.cohen_kappa_score,
    ]

    def __init__(self):
        super().__init__()
        self.results = None
        self.confusion_matrix = None
        self.mean_per_class_accuracy = None

    def compute_metrics(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        process_time: float = None):
        """
        Compute all metrics on predicted labels.

        :param y_true: Labels as a one-dimensional numpy array.
        :param y_pred: Model's predictions as a one-dimensional numpy array.
        :param process_time: Time of the process.
        """
        self.results = {metric_function.__name__: metric_function(
            y_true, y_pred) for metric_function in self.METRICS}
        if process_time is not None:
            self.results['process_time'] = process_time
        self.confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        self.mean_per_class_accuracy = \
            self.confusion_matrix.diagonal() /\
            self.confusion_matrix.sum(axis=1)
        return self

    def save_metrics(self, dest_path: str):
        """
        Save all metrics calculated on model's predictions.

        :param dest_path: Destination path to save the metrics. 
        """
        with open(os.path.join(dest_path, 'inference_metrics.csv'), 'w') as csv_file:
            writer = csv.DictWriter(csv_file, self.results.keys())
            writer.writeheader()
            writer.writerow(self.results)

        np.savetxt(os.path.join(dest_path, 'mean_per_class_accuracy.csv'),
                   self.mean_per_class_accuracy, delimiter=',', fmt='%.3f')

        np.savetxt(os.path.join(dest_path, 'confusion_matrix.csv'),
                   self.confusion_matrix, delimiter=',', fmt='%d')

    @staticmethod
    def timeit(function):
        """
        Time passed function as a decorator.
        """
        def timed(*args, **kwargs):
            start = time()
            result = function(*args, **kwargs)
            stop = time()
            return result, stop-start
        return timed
