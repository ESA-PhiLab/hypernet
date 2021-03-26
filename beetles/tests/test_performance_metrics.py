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
            (np.random.randint(0, 2, 10), np.random.randint(0, 2, 10))
        ])
    def test_mean_per_class_accuracy(self, y_true, y_pred):
        result = mean_per_class_accuracy(y_true=y_true, y_pred=y_pred)
        assert (result.shape[0] - 1) == np.max(y_true) == np.max(
            y_pred), 'Check is the nmber of classes is correct.'
        average_acc = metrics.balanced_accuracy_score(y_true, y_pred)
        assert np.mean(
            result) == average_acc, 'Average accuracy should be correct.'

    @pytest.mark.parametrize(
        'y_true, y_pred, metric_list, n_classes',
        [
            (np.arange(10), np.arange(10),
             [metrics.balanced_accuracy_score,
              metrics.accuracy_score,
              metrics.confusion_matrix,
              metrics.cohen_kappa_score],
             10),
            (np.random.permutation(25), np.random.permutation(25),
             [metrics.balanced_accuracy_score,
              metrics.accuracy_score,
              metrics.confusion_matrix,
              metrics.cohen_kappa_score],
             25)
        ]
    )
    def test_compute_metrics(self, y_true, y_pred,
                             metric_list, n_classes: int):
        result = compute_metrics(y_true, y_pred, metric_list)
        oa = float(np.sum(y_true == y_pred) / y_true.shape[0])
        assert oa == result[metrics.accuracy_score.__name__][0], \
            'The overall accuracy should be correct.'
        conf_matrix = np.zeros((n_classes, n_classes))
        for i in range(len(y_true)):
            conf_matrix[y_true[i], y_pred[i]] += 1
        np.testing.assert_array_equal(
            conf_matrix,
            result[metrics.confusion_matrix.__name__][0],
            'The confusion matrix must be correct.')
        per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        assert np.mean(per_class_acc) == \
            result[metrics.balanced_accuracy_score.__name__][0], \
            'The average accuracy should be correct.'
