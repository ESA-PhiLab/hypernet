import csv
import os
from typing import Dict, Union

import numpy as np
from sklearn import metrics


class Metrics:
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

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.results = {metric_function.__name__: metric_function(
            y_true, y_pred) for metric_function in self.METRICS}
        self.confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        self.mean_per_class_accuracy = \
            self.confusion_matrix.diagonal() /\
            self.confusion_matrix.sum(axis=1)
        return self

    def save_metrics(self, dest_path: str):
        with open(os.path.join(dest_path, 'metrics.csv'), 'w') as file:
            writer = csv.DictWriter(file, self.results.keys())
            writer.writeheader()
            writer.writerow(self.results)
        np.savetxt(os.path.join(dest_path, 'mean_per_class_accuracy.csv'),
                   self.mean_per_class_accuracy, delimiter=',')
        np.savetxt(os.path.join(dest_path, 'confusion_matrix.csv'),
                   self.confusion_matrix, delimiter=',')
