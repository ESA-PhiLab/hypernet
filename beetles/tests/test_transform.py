import numpy as np
import pytest

from ml_intuition.data import transforms


class TestSpectralTransform:
    @pytest.mark.parametrize("sample, label",
                             [(np.arange(10),
                               np.arange(10)),
                              ])
    def test_if_returns_expanded_tensors(self,
                                         sample: np.ndarray,
                                         label: np.ndarray):
        sample, label = transforms.SpectralTransform()(sample, label)
        assert sample.shape[-1] == 1, 'Assert correct expanded dims.'


class TestOneHotEncode:
    @pytest.mark.parametrize("sample, label, n_classes",
                             [(np.arange(10),
                               np.arange(10),
                               10),
                              (np.random.permutation(
                                  np.arange(10).reshape((5, -1))),
                               np.random.permutation(np.arange(5)),
                               5),
                              ])
    def test_if_one_hot_encoded_correctly(self,
                                          sample: np.ndarray,
                                          label: np.ndarray,
                                          n_classes: int):
        _, tr_label = transforms.OneHotEncode(n_classes)(sample, label)
        assert len(tr_label.shape) == 2
        one_hot = np.zeros((label.size, label.max()+1))
        one_hot[np.arange(label.size), label] = 1
        np.testing.assert_array_equal(one_hot, tr_label)


class TestMinMaxNormalize:
    @pytest.mark.parametrize("sample, label",
                             [
                                 (np.random.permutation(
                                     np.arange(10).reshape((5, -1))),
                                  np.random.permutation(
                                      np.arange(5))),
                             ])
    def test_if_normalize_correct(self,
                                  sample: np.ndarray,
                                  label: np.ndarray):
        tr_sample, _ = transforms.MinMaxNormalize(
            np.amin(sample), np.amax(sample))(sample, label)
        assert np.amax(tr_sample) == 1
        assert np.amin(tr_sample) == 0
