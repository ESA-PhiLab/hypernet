import abc
import numpy as np
from typing import List
from random import randint
from sklearn.decomposition import PCA
import skimage.transform as transform

from python_research.dataset_structures import Dataset

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


class StdDevNoiseTransformation(ITransformation):
    """
    Calculate standard deviation for each class' band, and use this standard
    deviation to draw a random value from a distribution (with mean=0),
    multiply it by a provided constant alpha and add this value to pixel's
    band original one. Number of transformation of a single pixel is equal to
    the number of unique values of alpha parameter.
    """
    def __init__(self, alphas=None, concatenate: bool=True,
                 mode: str = 'per_band'):
        """
        :param alphas: Scaling value of a random value
        :param concatenate: Whether to add transformed data to the original one,
                            or return a new dataset with transformed data only
        :param mode: Indicates whether standard deviation should be calculated
             globally or for each band independently.
        """
        if alphas is None:
            self.alphas = [0.1, 0.9]
        else:
            self.alphas = alphas
        self.concatenate = concatenate
        self.std_dev = None
        self.mode = mode

    def fit(self, data: np.ndarray) -> None:
        """

        :param data: Data to fit to

        """
        if self.mode == 'per_band':
            self.std_dev = self._collect_stddevs_per_band(data)
        elif self.mode == 'globally':
            self.std_dev = self._collect_stddevs_globally(data)
        else:
            raise ValueError("Mode {} is not implemented".format(self.mode))

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
            std_dev = np.std(dataset[..., band])
            std_devs.append(std_dev)
        return std_devs

    @staticmethod
    def _collect_stddevs_globally(dataset: Dataset) -> float:
        """
        Calculate standard deviation for the whole dataset
        :param dataset: Dataset to calculate standard deviation for
        :return: Standrad deviation for the whole dataset
        """
        return np.std(dataset)

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


class RandomRotationTransform(ITransformation):
    """
    Transformation which rotates a given sample by some multiple of 90 degrees.
    """
    def __init__(self):
        super(RandomRotationTransform, self).__init__()

    def transform(self, data: np.ndarray, rotations: int=1):
        """
        Transform samples
        :param data: Data to be transformed
        :param rotations: Number of times that each sample will be rotated by
        90 degrees. For example, 1 means rotation by 90 degrees, 2 by 180 etc.
        :return: np.ndarray with rotated samples
        """
        rotated = np.zeros(data.shape)
        for sample in range(data.shape[SAMPLES_COUNT]):
            rotation = randint(1, 3)
            rotated[sample] = np.rot90(data[sample], rotation, axes=[0, 1])
        return rotated

    def fit(self, data: np.ndarray):
        pass


class RandomFlipTransform(ITransformation):
    """
    Transformation which flips the sample verticaly or horizontaly at random.
    """
    def __init__(self):
        super(RandomFlipTransform, self).__init__()

    def transform(self, data: np.ndarray,
                  transformations: int=1):
        """
        Transform samples
        :param data: Data to be transformed
        :param transformations_count: Number of transformations for each class
        :return:
        """
        flipped = np.zeros(data.shape)
        for sample in range(data.shape[SAMPLES_COUNT]):
            flip_orientation = randint(1, 2)
            flipped[sample] = np.flip(data[sample], flip_orientation)
        return flipped

    def fit(self, data: np.ndarray):
        pass


class UpScaleTransform(ITransformation):
    """
    Transformation which upscales a given sample, and then crops it to the
    original size.
    """
    def __init__(self):
        super(UpScaleTransform, self).__init__()

    @staticmethod
    def _crop_center(image: np.ndarray, x_size, y_size):
        y, x = image.shape[0:2]
        startx = x // 2 - (x_size // 2)
        starty = y // 2 - (y_size // 2)
        return image[starty:starty + y_size, startx:startx + x_size, ...]

    def transform(self, data: np.ndarray,
                  scale: float=1.25):
        """
        Transform samples
        :param data: Data to be transformed
        :param scale: Rescaling factor
        :return:
        """
        original_size_x, original_size_y = data.shape[1:3]
        scaled = np.zeros(data.shape)
        for sample in range(data.shape[SAMPLES_COUNT]):
            scaled_sample = transform.rescale(data[sample], scale=scale)
            scaled[sample] = self._crop_center(scaled_sample, original_size_x,
                                               original_size_y)
        return scaled

    def fit(self, data: np.ndarray):
        pass


class RandomBasicTransform(ITransformation):
    """
    Transformation which picks a transformation to use randomly between the
    following: Rotation, Flip, Upscaling.
    """
    def __init__(self):
        self.transformations = [RandomRotationTransform(),
                                RandomFlipTransform(),
                                UpScaleTransform()]
        super(RandomBasicTransform, self).__init__()

    def transform(self, data: np.ndarray,
                  transformations: int=1):
        """
        Transform samples
        :param data: Data to be transformed
        :param transformations_count: Number of transformations for each class
        :return:
        """
        transformed = np.zeros(data.shape)
        for sample in range(data.shape[SAMPLES_COUNT]):
            random_transformation = randint(0, len(self.transformations) - 1)
            size_one_batch = np.expand_dims(data[sample], axis=0)
            transformed[sample] = self.transformations[random_transformation].transform(size_one_batch)[0, ...]
        return transformed

    def fit(self, data: np.ndarray):
        pass


class LightenTransform(ITransformation):
    """
    Transformation which adds a mean band average from respective bands,
    scaled by a scaling factor.
    """
    def __init__(self):
        self.per_band_average = None
        super(LightenTransform, self).__init__()

    def fit(self, data: np.ndarray):
        self.per_band_average = np.average(data, axis=0)

    def transform(self, data: np.ndarray,
                  transformations: int=1, scaling: float=0.1):
        """
        Transform samples
        :param data: Data to be transformed
        :param transformations_count: Number of transformations for each class
        :param scaling: Average scaling factor
        :return:
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        transformed = np.zeros(data.shape)
        for sample in range(data.shape[SAMPLES_COUNT]):
            transformed[sample] = data[sample] + (self.per_band_average *
                                                  scaling)
        return transformed


class DarkenTransform(ITransformation):
    """
    Transformation which subtracts a mean band average from respective bands,
    scaled by a scaling factor.
    """
    def __init__(self):
        self.per_band_average = None
        super(DarkenTransform, self).__init__()

    def fit(self, data: np.ndarray):
        self.per_band_average = np.average(data, axis=0)

    def transform(self, data: np.ndarray,
                  transformations: int=1, scaling: float=0.1):
        """
        Transform samples
        :param data: Data to be transformed
        :param transformations_count: Number of transformations for each class
        :param scaling: Average scaling factor
        :return:
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        transformed = np.zeros(data.shape)
        for sample in range(data.shape[SAMPLES_COUNT]):
            transformed[sample] = data[sample] - (self.per_band_average *
                                                  scaling)
        return transformed


class OnlineLightenTransform(ITransformation):
    """
    Transformation which adds a mean band average from respective bands,
    scaled by a scaling factor. Used for online augmentation.
    """
    def __init__(self, scaling: List[float]):
        self.per_band_average = None
        self.scaling = scaling
        super(OnlineLightenTransform, self).__init__()

    def fit(self, data: np.ndarray):
        self.per_band_average = np.average(data, axis=0)

    def transform(self, data: np.ndarray,
                  transformations: int=1):
        """
        Transform samples
        :param data: Data to be transformed
        :param transformations_count: Number of transformations for each class
        :return:
        """
        transformed = np.zeros((len(self.scaling) * 2, ) + data.shape)
        index = 0
        for scale in self.scaling:
            transformed[index] = data + (self.per_band_average * scale)
            transformed[index + 1] = data - (self.per_band_average * scale)
            index += len(self.scaling)
        return transformed
