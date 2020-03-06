import numpy as np
import pytest

import ml_intuition.data.preprocessing as preprocessing


class TestNormalizeLabels:
    @pytest.mark.parametrize("labels, correct_min, correct_max",
                             [(np.arange(2, 10), 0, 7),
                              (np.arange(0, 5), 0, 4),
                              (np.arange(7, 9), 0, 1)])
    def test_if_returns_correctly_normalized_labels(self, labels, correct_min,
                                                    correct_max):
        normalized = preprocessing.normalize_labels(labels)
        assert np.amin(normalized) == correct_min and np.amax(
            normalized) == correct_max


class TestReshapeTo2DSamples:

    @pytest.mark.parametrize(
        "input_shape, output_shape, labels_shape, channels_idx", [
            ((10, 10, 3), (100, 3, 1), (10, 10), 2),
            ((3, 5, 5), (25, 3, 1), (5, 5), 0),
            ((5, 3, 1), (15, 1, 1), (5, 3), 2)
        ])
    def test_if_reshapes_correctly(self, input_shape, output_shape,
                                   labels_shape, channels_idx):
        data = np.zeros(input_shape)
        labels = np.zeros(labels_shape)
        reshaped_data, _ = preprocessing.reshape_cube_to_2d_samples(data,
                                                                    labels,
                                                                    channels_idx)
        assert np.all(np.equal(reshaped_data.shape, output_shape))

    @pytest.mark.parametrize("data, channels_idx", [
        (np.arange(25).reshape((5, 5, 1)), 2),
        (np.arange(25).reshape((1, 5, 5)), 0)
    ])
    def test_if_data_matches_labels_after_reshape(self, data, channels_idx):
        labels = np.arange(25).reshape((5, 5))
        reshaped_data, reshaped_labels = preprocessing.reshape_cube_to_2d_samples(
            data, labels, channels_idx=channels_idx)
        assert np.all(np.equal(reshaped_data[:, 0, 0], reshaped_labels))


class TestRemoveNanSamples:

    def test_if_removes_correct_amount(self):
        data = np.zeros((3, 3, 1))
        labels = np.zeros(3)
        data[0, 0] = np.nan
        data[2, ...] = np.nan
        data, labels = preprocessing.remove_nan_samples(data, labels)
        assert len(data) == 1 and len(labels) == 1

    def test_if_returns_same_shape(self):
        data = np.zeros((3, 3, 5, 1))
        labels = np.zeros(3)
        data[0, ...] = np.nan
        data_n, labels_n = preprocessing.remove_nan_samples(data, labels)
        assert np.all(np.equal(data.shape[1:], data_n.shape[1:]))
