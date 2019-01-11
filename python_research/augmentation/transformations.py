import abc
import numpy as np
from typing import List
from sklearn.decomposition import PCA

from python_research.experiments.utils.datasets.hyperspectral_dataset import Dataset

SAMPLES_COUNT = 0


class ITransformation(abc.ABC):
    """
    Interface for transformations applied to Dataset
    """

    @abc.abstractmethod
    def transform(self, data: np.ndarray,
                  transformations: int=1) -> np.ndarray:
        """"
        Implements the transformation
        :param data: Numpy array of shape (batch, channels)
        :param transformations: Number of transformations of each sample
        """

    @abc.abstractmethod
    def fit(self, data: np.ndarray):
        """
        Implements fitting to the data before performing transformations
        :param data: Data to fit to
        """


class StdDevNoiseTransformation(ITransformation):
    """
    Calculate standard deviation for each class' band, and use this standard
    deviation to draw a random value from a distribution (with mean=0),
    multiply it by a provided constant alpha and add this value to pixel's
    band original one. Number of transformation of a single pixel is equal to
    the number of unique values of alpha parameter.
    """
    def __init__(self, alphas=None, concatenate: bool=True):
        """
        :param alphas: Scaling value of a random value
        :param concatenate: Whether to add transformed data to the original one,
                            or return a new dataset with transformed data only
        """
        if alphas is None:
            self.alphas = [0.1, 0.9]
        else:
            self.alphas = alphas
        self.concatenate = concatenate
        self.std_dev = None
        self.mode = None

    def fit(self, data: np.ndarray, mode: str='per_band') -> None:
        """

        :param data: Data to fit to
        :param mode: Indicates whether standard deviation should be calculated
                     globally or for each band independently.
        """
        if self.mode == 'per_band':
            self.std_dev = self._collect_stddevs_per_band(data)
        elif self.mode == 'globally':
            self.std_dev = self._collect_stddevs_globally(data)
        else:
            raise ValueError("Mode {} is not implemented".format(mode))

    @staticmethod
    def _collect_stddevs_per_band(dataset: Dataset) -> List[float]:
        """
        Calculate standard deviation for each band
        :param dataset: Dataset to calculate standard deviations for
        :return: List of standard deviations for each band respectively
        """
        bands = dataset.shape[-1]
        std_devs = list()
        for band in range(bands):
            std_dev = np.std(dataset.get_data()[..., band])
            std_devs.append(std_dev)
        return std_devs

    @staticmethod
    def _collect_stddevs_globally(dataset: Dataset) -> float:
        """
        Calculate standard deviation for the whole dataset
        :param dataset: Dataset to calculate standard deviation for
        :return: Standrad deviation for the whole dataset
        """
        return np.std(dataset.get_data())

    def _transform_globally(self, data: np.ndarray) -> np.ndarray:
        """
        Transform samples using global standard deviation
        :param data: Data to be transformed
        :return: Transformed data
        """
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        samples_count = len(data)
        augmented_data = list()
        for sample_index in range(samples_count):
            original_sample = data[sample_index]
            random_value = np.random.normal(loc=0, scale=self.std_dev)
            random_values = np.full(len(original_sample), random_value)
            for alpha in self.alphas:
                transformed_sample = original_sample + alpha * random_values
                augmented_data.append(transformed_sample)
        return np.array(augmented_data).astype(np.float64)

    def _transform_per_band(self, data: np.ndarray) -> np.ndarray:
        """
        Transforma data using standard deviation of each band
        :param data: Data to be transformed
        :return: Transformed data
        """
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        samples_count = len(data)
        augmented_data = list()
        for sample_index in range(samples_count):
            original_sample = data[sample_index]
            for alpha in self.alphas:
                random_values = np.random.normal(loc=0,
                                                 scale=self.std_dev)
                transformed_sample = original_sample + alpha * random_values
                augmented_data.append(transformed_sample)
        return np.array(augmented_data).astype(np.float64)

    def transform(self, data: np.ndarray, transformations: int=1) -> np.ndarray:
        """
        Perform the transformation based on previously selected mode
        :param data: Data to be transformed
        :param transformations: Number of transformations for each samples
        :return: Transformed data
        """
        if self.mode == 'per_band':
            return self._transform_per_band(data)
        elif self.mode == 'globally':
            return self._transform_globally(data)


class PCATransformation(ITransformation):
    """
    Transform samples using PCA, modify first component by multiplying
    it by a random value from a given range and then inverse
    transform principal components back to the original domain.
    """
    def __init__(self, n_components: float=2, low=0.9, high=1.1):
        """
        :param n_components: Number of components to be returned by PCA
        transformation
        :param low: Lower boundary of the random value range
        :param high: Upper boundary of the random value range
        """
        self.pca = PCA(n_components=n_components)
        self.low = low
        self.high = high

    def transform(self, data: np.ndarray, transformations_count: int=4) \
            -> np.ndarray:
        """
        Transform samples
        :param data: Data to be transformed
        :param transformations_count: Number of transformations for each class
        :return:
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        transformed = self.pca.transform(data)
        random_values = np.random.uniform(low=self.low, high=self.high,
                                          size=transformations_count *
                                               transformed.shape[SAMPLES_COUNT])
        transformed = np.repeat(transformed, transformations_count,
                                axis=0)
        transformed[:, 0] *= random_values
        return self.pca.inverse_transform(transformed)

    def fit(self, data: np.ndarray) -> None:
        """
        Fit PCA to data
        :param data: Data to fit to
        :return: None
        """
        self.pca = self.pca.fit(data)
