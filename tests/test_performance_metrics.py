import numpy as np
import pytest
from sklearn import metrics

from ml_intuition.evaluation.performance_metrics import (
    compute_metrics, mean_per_class_accuracy)


class TestPerformanceMetrics:
    @pytest.mark.parametrize(
        'y_true, y_pred',
        [
            (np.arange(10), np.arange(10)),
            (np.random.randint(0, 2, 10), np.random.randint(0, 2, 10)),
            pytest.param(np.arange(10).reshape(5, -1),
                         np.arange(10).reshape(5, -1), marks=pytest.mark.xfail)
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
            (np.arange(10), np.arange(10),
             [metrics.balanced_accuracy_score,
              metrics.accuracy_score,
              metrics.confusion_matrix,
              metrics.cohen_kappa_score]),
            (np.random.randint(0, 10, 10), np.random.randint(0, 10, 10),
             [metrics.balanced_accuracy_score,
              metrics.accuracy_score,
              metrics.confusion_matrix,
              metrics.cohen_kappa_score]),
            (np.full(10, 1), np.full(10, 1),
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
