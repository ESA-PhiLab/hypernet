import numpy as np
import pytest
import tensorflow as tf

from ml_intuition.data import transforms


class TestSpectralTransform:
    @pytest.mark.parametrize("sample, label",
                             [(tf.convert_to_tensor(np.arange(10)),
                               tf.convert_to_tensor(np.arange(10))),
                              ])
    def test_if_returns_expanded_tensors(self,
                                         sample: tf.Tensor,
                                         label: tf.Tensor):
        sample, label = transforms.SpectralTransform()(sample, label)
        assert sample.get_shape()[-1] == 1, 'Assert correct expanded dims.'
        assert sample.dtype == tf.float32, 'Assert type casting was correct.'


class TestOneHotEncode:
    @pytest.mark.parametrize("sample, label, n_classes",
                             [(tf.convert_to_tensor(np.arange(10)),
                               tf.convert_to_tensor(np.arange(10)),
                               10),
                              (tf.convert_to_tensor(
                                  np.random.permutation(
                                      np.arange(10).reshape((5, -1)))),
                               tf.convert_to_tensor(
                                  np.random.permutation(
                                      np.arange(5))),
                               5),
                              ])
    def test_if_one_hot_encoded_correctly(self,
                                          sample: tf.Tensor,
                                          label: tf.Tensor,
                                          n_classes: int):
        sess = tf.Session()
        npy_label = label.eval(session=sess)
        sample, label = transforms.OneHotEncode(n_classes)(sample, label)
        assert len(label.get_shape()) == 2
        one_hot_npy_label = np.zeros((npy_label.size, npy_label.max()+1))
        one_hot_npy_label[np.arange(npy_label.size), npy_label] = 1
        label = label.eval(session=sess)
        np.testing.assert_array_equal(one_hot_npy_label, label)


class TestMinMaxNormalize:
    @pytest.mark.parametrize("sample, label",
                             [
                                 (tf.cast(tf.convert_to_tensor(
                                  np.random.permutation(
                                      np.arange(10).reshape((5, -1)))), tf.float32),
                                  tf.convert_to_tensor(
                                  np.random.permutation(
                                      np.arange(5)))),
                             ])
    def test_if_normalize_correct(self,

                                  sample: tf.Tensor,
                                  label: tf.Tensor):
        sess = tf.Session()
        npy_sample = sample.eval(session=sess)
        sample, label = transforms.MinMaxNormalize(
            np.amin(npy_sample), np.amax(npy_sample))(sample, label)
        npy_sample = sample.eval(session=sess)
        assert np.amax(npy_sample) == 1
        assert np.amin(npy_sample) == 0
