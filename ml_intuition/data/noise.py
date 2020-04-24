import abc
import json
import math
import sys
from itertools import product
from typing import Dict, List

import numpy as np
import tensorflow as tf

from ml_intuition.enums import Dataset, Sample


class BaseNoise(abc.ABC):
    def __init__(self, params: Dict):
        super().__init__()
        self.params = params

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Each subclass should implement this method.

        :param args: Arbitrary list of arguments.
        :param kwargs: Arbitrary dictionary of arguments.
        """

    def get_proba(self, n_samples: int, prob: float) -> int:
        return math.floor(n_samples * prob)


class Gaussian(BaseNoise):

    def __call__(self, data: np.ndarray, labels: np.ndarray) -> List[np.ndarray]:
        """
        Perform Gaussian noise injection.

        :param data: Input data that will undergo noise injection.
        :param label: Class value for each data.
        :return: List containing the noisy data and the class label.
        """
        n_affected = self.get_proba(
            data.shape[Sample.SAMPLES_DIM], self.params['pa'])
        data = data.astype(np.float)
        noise = np.random.normal(
            loc=self.params['mean'], scale=self.params['std'],
            size=(n_affected, *data.shape[Sample.FEATURES_DIM:]))
        for noise_index, sample_index in enumerate(
                np.random.choice(data.shape[Sample.SAMPLES_DIM], n_affected, False)):
            data[sample_index] += noise[noise_index]
        return [data, labels]


class Impulsive(BaseNoise):

    def __call__(self, data: np.ndarray, labels: np.ndarray) -> List[np.ndarray]:
        """
        Perform impulsive noise injection.

        :param data: Input data that will undergo noise injection.
        :param label: Class value for each data.
        :return: List containing the noisy data and the class label.
        """
        n_affected = self.get_proba(
            data.shape[Sample.SAMPLES_DIM], self.params['pa'])
        n_white = self.get_proba(n_affected, self.params['pw'])

        black, white = np.amin(data), np.amax(data)
        affected_samples = np.random.choice(
            data.shape[Sample.SAMPLES_DIM], n_affected, False)

        for sample_index in affected_samples[:n_white]:
            data[sample_index] = np.full(
                shape=data[sample_index].shape, fill_value=white)
        for sample_index in affected_samples[n_white:]:
            data[sample_index] = np.full(
                shape=data[sample_index].shape, fill_value=black)
        return [data, labels]


def get_all_noise_functions(noise: str) -> List:
    """
    Get a given noise function.

    :param noise: Noise method as string.
    """
    all_ = {
        str(f).lower(): eval(f) for f in dir(sys.modules[__name__])
    }
    return [all_[noise_fun] for noise_fun in noise]


def get_noise_functions(noise: List[str], noise_params: str) -> List[BaseNoise]:
    return [noise_injector(json.loads(noise_params))
            for noise_injector in get_all_noise_functions(noise)]


def inject_noise(data_source: Dict, affected_subsets: List[str], noise_injectors: List[str], noise_params: str):
    for f_noise, affected_subset in product(
            get_noise_functions(noise_injectors, noise_params), affected_subsets):
        data_source[affected_subset][Dataset.DATA], data_source[affected_subset][Dataset.LABELS] = \
            f_noise(data_source[affected_subset][Dataset.DATA],
                    data_source[affected_subset][Dataset.LABELS])
