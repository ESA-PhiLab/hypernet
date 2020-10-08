import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from ml_intuition.evaluation.performance_metrics import overall_rms_abundance_angle_distance, \
    cnn_rmse, per_class_rmse, dcae_rmse, average_angle_spectral_mapper

sess = tf.Session()


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class TestRMSAbundanceAngleDistance:
    @pytest.mark.parametrize(
        'y_true, y_pred',
        [
            (np.random.uniform(0, 1, (10, 10)), np.random.uniform(0, 1, (10, 10))),
            (np.random.uniform(0, 1, (10, 2)), np.random.uniform(0, 1, (10, 2))),
            (np.random.uniform(0, 1, (10, 1)), np.random.uniform(0, 1, (10, 1)))
        ])
    def test_rmsaad_with_external_implementation(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_true, y_pred = softmax(y_true), softmax(y_pred)
        tf_y_true, tf_y_pred = tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)
        tf_error = overall_rms_abundance_angle_distance(y_true=tf_y_true, y_pred=tf_y_pred).eval(session=sess)
        error = 0
        for i in range(y_true.shape[0]):
            error += (np.arccos(np.dot(y_true[i], y_pred[i]) /
                                (np.linalg.norm(y_true[i]) * np.linalg.norm(y_pred[i])))) ** 2
        error /= y_true.shape[0]
        error = np.sqrt(error)
        assert round(tf_error, 3) == round(error, 3)


class TestRMSE:
    @pytest.mark.parametrize(
        'y_true, y_pred',
        [
            (np.random.uniform(0, 1, (10, 10)), np.random.uniform(0, 1, (10, 10))),
            (np.random.uniform(0, 1, (10, 2)), np.random.uniform(0, 1, (10, 2))),
            (np.random.uniform(0, 1, (10, 1)), np.random.uniform(0, 1, (10, 1)))
        ])
    def test_cnn_rmse_with_external_implementations(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_true, y_pred = softmax(y_true), softmax(y_pred)
        tf_y_true, tf_y_pred = tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)
        tf_overall_error = cnn_rmse(y_true=tf_y_true, y_pred=tf_y_pred).eval(session=sess)
        error = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert round(tf_overall_error, 3) == round(error, 3) == \
               round(np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred)), 3)
        tf_per_class_error = per_class_rmse(y_true=tf_y_true, y_pred=tf_y_pred).eval(session=sess)
        per_class_error = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
        assert np.array_equal(np.round(tf_per_class_error, 3), np.round(per_class_error, 3))

    @pytest.mark.parametrize(
        'y_true, y_pred',
        [
            (np.random.uniform(0, 1, (10, 10)), np.random.uniform(0, 1, (10, 10))),
            (np.random.uniform(0, 1, (10, 2)), np.random.uniform(0, 1, (10, 2))),
            (np.random.uniform(0, 1, (10, 1)), np.random.uniform(0, 1, (10, 1)))
        ])
    def test_dcae_rmse_with_external_implementations(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_true, y_pred = softmax(y_true), softmax(y_pred)
        tf_y_true, tf_y_pred = tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)
        tf_overall_error = dcae_rmse(y_true=tf_y_true, y_pred=tf_y_pred).eval(session=sess)
        error = np.mean(np.sqrt(np.mean(np.square(y_pred - y_true), axis=1)))
        assert round(tf_overall_error, 3) == round(error, 3)


class TestAverageAngleSpectralMapper:
    @pytest.mark.parametrize(
        'y_true, y_pred',
        [
            (np.random.uniform(0, 1, (10, 10)), np.random.uniform(0, 1, (10, 10))),
            (np.random.uniform(0, 1, (10, 2)), np.random.uniform(0, 1, (10, 2))),
            (np.random.uniform(0, 1, (10, 1)), np.random.uniform(0, 1, (10, 1)))
        ])
    def test_aSAM_with_external_implementations(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_true, y_pred = softmax(y_true), softmax(y_pred)
        tf_y_true, tf_y_pred = tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)
        tf_error = average_angle_spectral_mapper(y_true=tf_y_true, y_pred=tf_y_pred).eval(session=sess)
        error = 0
        for i in range(y_true.shape[0]):
            error += np.arccos(np.dot(y_true[i], y_pred[i]) / (np.linalg.norm(y_true[i]) * np.linalg.norm(y_pred[i])))
        error /= y_true.shape[0]
        assert round(tf_error, 3) == round(error, 3)
