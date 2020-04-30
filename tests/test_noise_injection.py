import math
from typing import Dict

import numpy as np
import pytest

from ml_intuition.data import noise


class TestGaussianNoise:
    @pytest.mark.parametrize("data, params",
                             [
                                 (np.random.rand(100, 20, 1), {
                                  "mean": 0, "std": 1, "pa": 0.1, "pb": 0.5, "bc": True}),
                                 (np.random.rand(250, 104, 1), {
                                  "mean": 5, "std": 10, "pa": 0.4, "pb": 0.9, "bc": False}),
                                 (np.random.rand(99, 1000), {
                                     "mean": 5, "std": 10, "pa": 0.96, "pb": 0.5, "bc": True})
                             ])
    def test_gaussian_noise_injection(self, data: np.ndarray, params: Dict):
        gaussian_noise = noise.Gaussian(params)
        data_prime, _ = gaussian_noise(data, None)
        assert not np.array_equal(data, data_prime)
        n_true_affected = gaussian_noise.get_proba(data.shape[0],
                                                   gaussian_noise.params.pa)
        n_true_bands = gaussian_noise.get_proba(data.shape[1],
                                                gaussian_noise.params.pb)
        n_affected = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.array_equal(data[i, j], data_prime[i, j]):
                    n_affected += 1
        assert n_affected == n_true_affected * n_true_bands, \
            'Assert the correct number of noise pixels and their bands.'

    @pytest.mark.parametrize("data, params",
                             [
                                 (np.random.rand(100, 20, 1), {
                                  "mean": 0, "std": 1, "pa": 1, "pb": 1, "bc": True}),
                                 (np.random.rand(160, 1, 4, 2, 1), {
                                  "mean": 5, "std": 10, "pa": 1, "pb": 1, "bc": False}),
                                 (np.random.rand(50, 1000), {
                                     "mean": 5, "std": 10, "pa": 1, "pb": 1, "bc": True})
                             ])
    def test_if_all_noise_injected(self, data: np.ndarray, params: Dict):
        gaussian_noise = noise.Gaussian(params)
        data_prime, _ = gaussian_noise(data, None)
        assert (data != data_prime).all(), \
            'Assert each element is augmented with noise (\"pa\" == 1)'

    @pytest.mark.parametrize("data, params",
                             [
                                 (np.random.rand(20, 20, 1), {
                                  "mean": 0, "std": 1, "pa": 0, "pb": 0, "bc": True}),
                                 (np.random.rand(76, 104), {
                                  "mean": 5, "std": 10, "pa": 0, "pb": 0.1, "bc": False}),
                                 (np.random.rand(34, 1000), {
                                     "mean": 5, "std": 10, "pa": 0, "pb": 0.5, "bc": True})
                             ])
    def test_if_no_noise_injected(self, data: np.ndarray, params: Dict):
        gaussian_noise = noise.Gaussian(params)
        data_prime, _ = gaussian_noise(data, None)
        assert (data == data_prime).all(), \
            'Assert no element is augmented with noise (\"pa\" == 0)'


class TestImpulsiveNoise:
    @pytest.mark.parametrize("data, params",
                             [
                                 (np.random.rand(20, 104, 1),
                                  {"pa": 0.5, "pw": 1, "pb": 0.5, "bc": True}),
                                 (np.random.rand(19, 20),
                                  {"pa": 0.5, "pw": 0.4, "pb": 0.5, "bc": False}),
                                 (np.random.rand(76, 20, 1),
                                  {"pa": 0.8, "pw": 0.1, "pb": 0.5, "bc": True})
                             ])
    def test_impulsive_nosie_injection(self, data: np.ndarray, params: Dict):
        impulsive_noise = noise.Impulsive(params)
        data_prime, _ = impulsive_noise(data, None)
        unique, counts = np.unique(data_prime, return_counts=True)
        n_affected, n_bands = \
            impulsive_noise.get_proba(data.shape[0],
                                      impulsive_noise.params.pa), \
            impulsive_noise.get_proba(data.shape[1],
                                      impulsive_noise.params.pb)
        n_white = impulsive_noise.get_proba(n_affected,
                                            impulsive_noise.params.pw)
        assert unique[-1] == np.amax(data), \
            'Assert correct value for white noise.'
        assert not abs(counts[-1] - math.floor(n_white * n_bands)) > 1, \
            'Assert the correct number of whitened pixels and their bands.'
        assert unique[0] == np.amin(data), \
            'Assert correct value for black noise.'
        assert not abs(counts[0] -
                       math.floor((n_affected - n_white) * n_bands)) > 1, \
            'Assert the correct number of blacked pixels and their bands.'

    @pytest.mark.parametrize("data, params",
                             [
                                 (np.random.rand(20, 20, 1),
                                  {"pa": 0, "pw": 0.2, "pb": 0.5, "bc": True}),
                                 (np.random.rand(76, 20, 1),
                                  {"pa": 0, "pw": 0.2, "pb": 0.5, "bc": True}),
                                 (np.random.rand(34, 1000, 1),
                                  {"pa": 0, "pw": 0.2, "pb": 0.5, "bc": True})
                             ])
    def test_if_no_noise_injected(self, data: np.ndarray, params: Dict):
        impulsive_noise = noise.Impulsive(params)
        data_prime, _ = impulsive_noise(data, None)
        assert (data == data_prime).all(), \
            'Assert no element is augmented with noise (\"pa\" == 0)'
