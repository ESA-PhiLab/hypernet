from typing import Dict

import numpy as np
import pytest

from ml_intuition.data import noise


class TestGaussianNoise:
    @pytest.mark.parametrize("data",
                             [
                                 (np.random.rand(100, 20, 1)),
                                 (np.random.rand(100, 1)),
                                 (np.random.rand(100, 1000))
                             ])
    def test_gaussian_noise_injection(self, data: np.ndarray):
        gaussian_noise = noise.Gaussian({"mean": 0, "std": 1, "pa": 0.5})
        data_prime, _ = gaussian_noise(data, None)
        assert not np.array_equal(data, data_prime)

    @pytest.mark.parametrize("data, params",
                             [
                                 (np.random.rand(100, 20, 1), {
                                  "mean": 0, "std": 1, "pa": 1}),
                                 (np.random.rand(100, 1, 4, 2, 1), {
                                  "mean": 5, "std": 10, "pa": 1}),
                                 (np.random.rand(100, 1000), {
                                     "mean": 5, "std": 10, "pa": 1})
                             ])
    def test_if_all_noise_injected(self, data: np.ndarray, params: Dict):
        gaussian_noise = noise.Gaussian(params)
        data_prime, _ = gaussian_noise(data, None)
        assert (data != data_prime).all(), \
            'Assert each element is augmented with noise (\"pa\" == 1)'

    @pytest.mark.parametrize("data, params",
                             [
                                 (np.random.rand(100, 20, 1), {
                                  "mean": 0, "std": 1, "pa": 0}),
                                 (np.random.rand(100, 1, 4, 2, 1), {
                                  "mean": 5, "std": 10, "pa": 0}),
                                 (np.random.rand(100, 1000), {
                                     "mean": 5, "std": 10, "pa": 0})
                             ])
    def test_if_no_noise_injected(self, data: np.ndarray, params: Dict):
        gaussian_noise = noise.Gaussian(params)
        data_prime, _ = gaussian_noise(data, None)
        assert (data == data_prime).all(), \
            'Assert no element is augmented with noise (\"pa\" == 0)'


class TestImpulsiveNoise:
    pass
