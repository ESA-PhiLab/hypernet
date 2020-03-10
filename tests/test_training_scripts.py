import os

import numpy as np
import pytest
import tensorflow as tf
from scripts import train_model
from scripts.models import get_model, model_2d
from sklearn import metrics

from ml_intuition.evaluation.performance_metrics import (
    compute_metrics, mean_per_class_accuracy)
from ml_intuition.evaluation.time_metrics import timeit


class TestMetrics:
    @pytest.mark.parametrize(
        'y_true, y_pred',
        [
            (np.arange(100), np.arange(100)),
            (np.random.randint(0, 10, 100), np.random.randint(0, 10, 100)),
            pytest.param(np.arange(100).reshape(10, -1),
                         np.arange(100).reshape(10, -1), marks=pytest.mark.xfail)
        ])
    def test_mean_per_class_accuracy(self, y_true, y_pred):
        result = mean_per_class_accuracy(y_true=y_true, y_pred=y_pred)
        assert result.ndim == 1, 'Mean per class accuracy should be a 1-dimensional vector.'
        assert (result.shape[0] - 1) == np.max(y_true) == np.max(
            y_pred), 'Number of classes should be consistent.'
        average_acc = metrics.balanced_accuracy_score(y_true, y_pred)
        assert np.mean(
            result) == average_acc, 'Average accuracy should be consistent.'

    @pytest.mark.parametrize(
        'y_true, y_pred, metric_list',
        [
            (np.arange(100), np.arange(100),
             [metrics.balanced_accuracy_score,
              metrics.accuracy_score,
              metrics.confusion_matrix,
              metrics.cohen_kappa_score]),
            (np.random.randint(0, 10, 100), np.random.randint(0, 10, 100),
             [metrics.balanced_accuracy_score,
              metrics.accuracy_score,
              metrics.confusion_matrix,
              metrics.cohen_kappa_score]),
            (np.full(100, 1), np.full(100, 1),
             [metrics.balanced_accuracy_score,
              metrics.accuracy_score,
              metrics.confusion_matrix,
              metrics.cohen_kappa_score])
        ]
    )
    def test_compute_metrics(self, y_true, y_pred, metric_list):
        result = compute_metrics(y_true, y_pred, metric_list)
        assert list(result.keys()) == \
            [metric_f.__name__ for metric_f in metric_list], 'All metrics should be present.'


class TestModels:
    @pytest.mark.parametrize(
        'model_key, kernel_size, n_kernels, n_layers, input_size, n_classes, lr',
        [
            ('model_2d', 4, 3, 2, 103, 9, 0.01),
            pytest.param('model_2d', 4, 30, 5, 103, 9,
                         0.01, marks=pytest.mark.xfail),
            pytest.param('model_2d', 4, -3, 5, -4, 9,
                         0.01, marks=pytest.mark.xfail),
            pytest.param('nonexisting_key_example', 4, 3, 1, 100, 9,
                         0.01, marks=pytest.mark.xfail),
        ]
    )
    def test_get_model(self, model_key, kernel_size, n_kernels, n_layers, input_size, n_classes, lr):
        model = get_model(model_key, kernel_size, n_kernels,
                          n_layers, input_size, n_classes, lr)
        assert isinstance(model, tf.keras.Sequential), 'Assert the model type.'
        layer = model.get_layer(index=2)
        assert isinstance(
            layer, tf.keras.layers.Layer), 'Assert not empyt model.'


class TestTimeMetrics:
    @pytest.mark.parametrize(
        'function',
        [
            (lambda: print('Example text.')),
            (lambda: print(str(elem) for elem in range(99)))
        ]
    )
    def test_timeit(self, function):
        function = timeit(function)
        _, result = function()
        assert type(result) == float, 'Assert the time type.'
