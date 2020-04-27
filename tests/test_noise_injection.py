import math
from typing import Dict

import numpy as np
import pytest

from ml_intuition.data import noise


class TestGaussianNoise:
    @pytest.mark.parametrize("data, params",
                             [
                                 (np.random.rand(100, 20, 1), {
                                  "mean": 0, "std": 1, "pa": 0.1}),
                                 (np.random.rand(250, 1, 4, 2, 1), {
                                  "mean": 5, "std": 10, "pa": 0.4}),
                                 (np.random.rand(99, 1000), {
                                     "mean": 5, "std": 10, "pa": 0.96})
                             ])
    def test_gaussian_noise_injection(self, data: np.ndarray, params: Dict):
        gaussian_noise = noise.Gaussian(params)
        data_prime, _ = gaussian_noise(data, None)
        assert not np.array_equal(data, data_prime)
        n_affected = 0
        for i in range(data.shape[0]):
            if not np.array_equal(data[i], data_prime[i]):
                n_affected += 1
        assert n_affected == math.floor(data.shape[0] * params['pa'])

    @pytest.mark.parametrize("data, params",
                             [
                                 (np.random.rand(100, 20, 1), {
                                  "mean": 0, "std": 1, "pa": 1}),
                                 (np.random.rand(160, 1, 4, 2, 1), {
                                  "mean": 5, "std": 10, "pa": 1}),
                                 (np.random.rand(50, 1000), {
                                     "mean": 5, "std": 10, "pa": 1})
                             ])
    def test_if_all_noise_injected(self, data: np.ndarray, params: Dict):
        gaussian_noise = noise.Gaussian(params)
        data_prime, _ = gaussian_noise(data, None)
        assert (data != data_prime).all(), \
            'Assert each element is augmented with noise (\"pa\" == 1)'

    @pytest.mark.parametrize("data, params",
                             [
                                 (np.random.rand(20, 20, 1), {
                                  "mean": 0, "std": 1, "pa": 0}),
                                 (np.random.rand(76, 1, 4, 2, 1), {
                                  "mean": 5, "std": 10, "pa": 0}),
                                 (np.random.rand(34, 1000), {
                                     "mean": 5, "std": 10, "pa": 0})
                             ])
    def test_if_no_noise_injected(self, data: np.ndarray, params: Dict):
        gaussian_noise = noise.Gaussian(params)
        data_prime, _ = gaussian_noise(data, None)
        assert (data == data_prime).all(), \
            'Assert no element is augmented with noise (\"pa\" == 0)'


class TestImpulsiveNoise:
    @pytest.mark.parametrize("data, params",
                             [
                                 (np.random.rand(20, 20, 5),
                                  {"mean": 0, "std": 1, "pa": 0.5, "pw": 1}),
                                 (np.random.rand(20, 20, 5),
                                  {"mean": 0, "std": 1, "pa": 0.5, "pw": 0.4}),
                                 (np.random.rand(76, 1, 5),
                                  {"mean": 0, "std": 1, "pa": 0.8, "pw": 0.1})
                             ])
    def test_impulsive_nosie_injection(self, data: np.ndarray, params: Dict):
        impulsive_noise = noise.Impulsive(params)
        data_prime, _ = impulsive_noise(data, None)
        unique, counts = np.unique(data_prime, return_counts=True)
        assert unique[-1] == np.amax(data), \
            'Assert correct value for white noise.'
        assert math.floor(counts[-1] / np.prod(data.shape[1:])) == \
            math.floor(data.shape[0] * params['pa'] * params['pw']), \
            'Assert the correct number of whitened pixels.'
        assert unique[0] == np.amin(data), \
            'Assert correct value for black noise.'
        assert math.floor(counts[0] / np.prod(data.shape[1:])) == \
            math.floor(data.shape[0] * params['pa'] * (1 - params['pw'])), \
            'Assert the correct number of blacked pixels.'

    @pytest.mark.parametrize("data, params",
                             [
                                 (np.random.rand(20, 20, 1), {
                                  "mean": 0, "std": 1, "pa": 0, "pw": 0.2}),
                                 (np.random.rand(76, 1, 4, 2, 1), {
                                  "mean": 5, "std": 10, "pa": 0, "pw": 0.2}),
                                 (np.random.rand(34, 1000), {
                                     "mean": 5, "std": 10, "pa": 0, "pw": 0.2})
                             ])
    def test_if_no_noise_injected(self, data: np.ndarray, params: Dict):
        impulsive_noise = noise.Impulsive(params)
        data_prime, _ = impulsive_noise(data, None)
        assert (data == data_prime).all(), \
            'Assert no element is augmented with noise (\"pa\" == 0)'
