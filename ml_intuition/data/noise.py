import abc
import json
import math
import sys
from itertools import product
from typing import Dict, List, NamedTuple

import numpy as np
import yaml

from ml_intuition.enums import Dataset, Sample


class Params(NamedTuple):
    """
    Parameters of noise injection.

    pa - Fraction of noisy pixels, the number of affected samples
    is calculated by: floor(n_samples * pa).

    pb - Fraction of noisy bands. When established the number of
    samples that undergo noise injection, for each sample
    the: floor(n_bands * pb) bands are affected.

    bc - Boolean indicating whether the indexes of affected bands
    are constant for each sample. When set to: False,
    different bands can be augmented with noise for each pixel.

    mean - Gaussian noise parameter, the mean of the normal distribution.

    std - Standard deviation of the normal distribution for Gaussian noise.

    pw - Ratio of whitened pixels for the affected set of samples
    in the Impulsive noise injection.
    """
    pa: float = None
    pb: float = None
    bc: bool = None
    mean: float = None
    std: float = None
    pw: float = None


class BaseNoise(abc.ABC):
    def __init__(self, params: Dict):
        super().__init__()
        default_params = {'pa': None, 'pb': None, 'bc': None, 'mean': None, 'std': None, 'pw': None}
        for key, value in params.items():
            default_params[key] = value
        self.params = Params(**default_params)

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Each subclass should implement this method.

        :param args: Arbitrary list of arguments.
        :param kwargs: Arbitrary dictionary of arguments.
        """

    @staticmethod
    def get_proba(n_samples: int, prob: float) -> int:
        return math.floor(n_samples * prob)


class Gaussian(BaseNoise):

    def __call__(self, data: np.ndarray, labels: np.ndarray) -> List[np.ndarray]:
        """
        Perform Gaussian noise injection.

        :param data: Input data that will undergo noise injection.
        :param labels: Class value for each data point.
        :return: List containing the noisy data and the class label.
        """
        n_affected, n_bands = \
            self.get_proba(data.shape[Sample.SAMPLES_DIM], self.params.pa), \
            self.get_proba(data.shape[Sample.FEATURES_DIM], self.params.pb)
        data = data.astype(np.float)
        noisy_bands = np.random.choice(data.shape[Sample.FEATURES_DIM],
                                       n_bands, False)
        for sample_index in np.random.choice(data.shape[Sample.SAMPLES_DIM],
                                             n_affected, False):
            if not self.params.bc:
                noisy_bands = np.random.choice(data.shape[Sample.FEATURES_DIM],
                                               n_bands, False)
            for band_index in noisy_bands:
                data[sample_index, band_index] += \
                    np.random.normal(loc=self.params.mean,
                                     scale=self.params.std,
                                     size=data[sample_index, band_index].shape)
        return [data, labels]


class Impulsive(BaseNoise):

    def __call__(self, data: np.ndarray, labels: np.ndarray) -> List[np.ndarray]:
        """
        Perform impulsive noise injection.

        :param data: Input data that will undergo noise injection.
        :param labels: Class value for each data point.
        :return: List containing the noisy data and the class label.
        """
        n_affected, n_bands = \
            self.get_proba(data.shape[Sample.SAMPLES_DIM], self.params.pa), \
            self.get_proba(data.shape[Sample.FEATURES_DIM], self.params.pb)
        n_white = self.get_proba(n_affected, self.params.pw)
        black, white = np.amin(data), np.amax(data)
        noisy_bands = np.random.choice(data.shape[Sample.FEATURES_DIM],
                                       n_bands, False)
        for noise_index, sample_index in enumerate(np.random.choice(
                data.shape[Sample.SAMPLES_DIM], n_affected, False)):
            if not self.params.bc:
                noisy_bands = np.random.choice(data.shape[Sample.FEATURES_DIM],
                                               n_bands, False)
            for band_index in noisy_bands:
                if noise_index < n_white:
                    data[sample_index, band_index] = \
                        np.full(shape=data[sample_index, band_index].shape,
                                fill_value=white)
                else:
                    data[sample_index, band_index] = \
                        np.full(shape=data[sample_index, band_index].shape,
                                fill_value=black)
        return [data, labels]


class Shot(BaseNoise):

    def __call__(self, data: np.ndarray, labels: np.ndarray) -> List[np.ndarray]:
        """
        Perform shot noise injection.

        :param data: Input data that will undergo noise injection.
        :param labels: Class value for each data point.
        :return: List containing the noisy data and the class label.
        """
        n_affected, n_bands = \
            self.get_proba(data.shape[Sample.SAMPLES_DIM], self.params.pa), \
            self.get_proba(data.shape[Sample.FEATURES_DIM], self.params.pb)
        data = data.astype(np.float)
        data = np.clip(data, a_min=0, a_max=None)
        noise = np.random.poisson(data)
        noisy_bands = np.random.choice(data.shape[Sample.FEATURES_DIM],
                                       n_bands, False)
        for sample_index in np.random.choice(data.shape[Sample.SAMPLES_DIM],
                                             n_affected, False):
            if not self.params.bc:
                noisy_bands = np.random.choice(data.shape[Sample.FEATURES_DIM],
                                               n_bands, False)
            for band_index in noisy_bands:
                data[sample_index, band_index] += noise[sample_index, band_index]
        return [data, labels]


def get_all_noise_functions(noise: str) -> List:
    """
    Get a given noise function.

    :param noise: Noise method as string.
    :return: List of all noise functions.
    """
    all_ = {
        str(f).lower(): eval(f) for f in dir(sys.modules[__name__])
    }
    return [all_[noise_fun] for noise_fun in noise]


def get_noise_functions(noise: List[str], noise_params: str) -> List[BaseNoise]:
    """
    Helper function for getting all noise injector instances.

    :param noise: List of noise injection methods.
    :param noise_params: Parameters of the noise injection.
    :return: List of instances of noise injectors.
    """
    try:
        noise_params = json.loads(noise_params)
    except json.decoder.JSONDecodeError:
        noise_params = yaml.load(noise_params)
    return [noise_injector(noise_params)
            for noise_injector in get_all_noise_functions(noise)]


def inject_noise(data_source: Dict, affected_subsets: List[str],
                 noise_injectors: List[str], noise_params: str):
    """
    Inject noise into given subsets.

    :param data_source: Dictionary containing subsets.
    :param affected_subsets: List of names of subsets that will undergo noise injection.
    :param noise_injectors: List of names of noise injection methods.
    :param noise_params: Parameters of the noise injection.
    """
    for f_noise, affected_subset in product(
            get_noise_functions(noise_injectors, noise_params), affected_subsets):
        data_source[affected_subset][Dataset.DATA], data_source[affected_subset][Dataset.LABELS] = f_noise(
            data_source[affected_subset][Dataset.DATA],
            data_source[affected_subset][Dataset.LABELS])
