"""
Module containing all the transformations that can be done on a dataset.
"""

import abc
from typing import List, Dict

import numpy as np

from ml_intuition import enums
from ml_intuition.models import unmixing_pixel_based_dcae, \
    unmixing_cube_based_dcae, unmixing_cube_based_cnn, \
    unmixing_pixel_based_cnn, unmixing_rnn_supervised


class BaseTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Each subclass should implement this method.

        :param args: Arbitrary list of arguments.
        :param kwargs: Arbitrary dictionary of arguments.
        """
        pass


class SpectralTransform(BaseTransform):
    def __init__(self, **kwargs):
        """
        Initializer of the spectral transformation.
        """
        super().__init__()

    def __call__(self, samples: np.ndarray,
                 labels: np.ndarray) -> List[np.ndarray]:
        """
        Transform 1D samples along the spectral axis.
        Only the spectral features are present for each sample in the dataset.

        :param samples: Input samples that will undergo transformation.
        :param labels: Class value for each samples.
        :return: List containing the transformed samples and the class labels.
        """
        return [np.expand_dims(samples.astype(np.float32), -1), labels]


class OneHotEncode(BaseTransform):
    def __init__(self, n_classes: int):
        """
        Initializer of the one-hot encoding transformation.

        :param n_classes: Number of classes.
        """
        super().__init__()
        self.n_classes = n_classes

    def __call__(self, samples: np.ndarray, labels: np.ndarray) -> List[
        np.ndarray]:
        """
        Perform one-hot encoding on the passed labels.

        :param samples: Input samples.
        :param labels: Class values for each sample
            that will undergo one-hot encoding.
        :return: List containing the samples
            and the one-hot encoded class labels.
        """
        out_label = np.zeros((labels.size, self.n_classes))
        out_label[np.arange(labels.size), labels] = 1
        return [samples, out_label.astype(np.uint8)]


class MinMaxNormalize(BaseTransform):
    def __init__(self, min_: float, max_: float):
        """
        Normalize each sample.

        :param min_: Minimum value of features.
        :param max_: Maximum value of features.
        """
        super().__init__()
        self.min_ = min_
        self.max_ = max_

    def __call__(self, samples: np.ndarray, labels: np.ndarray) -> List[
        np.ndarray]:
        """"
        Perform min-max normalization on the passed samples.

        :param samples: Input samples that will undergo normalization.
        :param labels: Class values for each sample.
        :return: List containing the normalized samples and the class labels.
        """
        return [(samples - self.min_) / (self.max_ - self.min_), labels]


def apply_transformations(data: Dict,
                          transformations: List[BaseTransform]) -> Dict:
    """
    Apply each transformation from provided list

    :param data: Dictionary with 'data' and 'labels' keys holding np.ndarrays
    :param transformations: List of transformations
    :return: Transformed data, in the same format as input
    """
    for transformation in transformations:
        data[enums.Dataset.DATA], data[enums.Dataset.LABELS] = transformation(
            data[enums.Dataset.DATA], data[enums.Dataset.LABELS])
    return data


class RNNSpectralInputTransform(BaseTransform):

    def __call__(self, samples: np.ndarray,
                 labels: np.ndarray) -> List[np.ndarray]:
        """"
        Transform the input samples to fit the recurrent
        neural network (RNN) input.
        This is performed for the pixel-based model;
        the input sample includes only spectral bands.

        :param samples: Input samples that will undergo the transformation.
        :param labels: Class values for each sample.
        :return: List containing the normalized samples and the class labels.
        """
        return [np.expand_dims(np.squeeze(samples), -1), labels]


class PerBandMinMaxNormalization(BaseTransform):
    SPECTRAL_DIM = -1

    def __init__(self, min_: np.ndarray, max_: np.ndarray):
        self.min_ = min_
        self.max_ = max_

    def __call__(self, samples: np.ndarray, labels: np.ndarray) -> List[
        np.ndarray]:
        """
        Perform per-band min-max normalization.
        Each band is treated as a separate feature.

        :param samples: Input samples that will undergo transformation.
        :param labels: Abundance vectors for each sample.
        :return: List containing the normalized samples and the abundance vectors.
        """
        samples, labels = samples.astype(np.float32), labels.astype(np.float32)
        sample_shape = samples.shape
        samples = samples.reshape(-1, samples.shape[
            PerBandMinMaxNormalization.SPECTRAL_DIM])
        for band_index, (min_val, max_val) in enumerate(
                zip(self.min_, self.max_)):
            samples[..., band_index] = (samples[..., band_index] - min_val) / (
                    max_val - min_val)
        return [samples.reshape(sample_shape), labels]

    @staticmethod
    def get_min_max_vectors(data_cube: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get the min-max vectors for each spectral band.

        :param data_cube: Hyperspectral data cube,
            with bands in the last dimension and
            spatial features in the first two dimensions.
        :return: Dictionary containing the min as well as
            the max vectors for each band.
        """
        data_cube = data_cube.reshape(-1, data_cube.shape[
            PerBandMinMaxNormalization.SPECTRAL_DIM])
        return {'min_': np.amin(data_cube, axis=0),
                'max_': np.amax(data_cube, axis=0)}


class ExtractCentralPixelSpectrumTransform(BaseTransform):
    def __init__(self, neighborhood_size: int):
        """
        Extract central pixel from each spatial sample.

        :param neighborhood_size: The spatial size of the patch.
        """
        super().__init__()
        self.neighborhood_size = neighborhood_size

    def __call__(self, samples: np.ndarray,
                 labels: np.ndarray) -> List[np.ndarray]:
        """"
        Transform the labels for unsupervised unmixing problem.
        The label is the central pixel of each sample patch.

        :param samples: Input samples.
        :param labels: Central pixel of each sample.
        :return: List containing the input samples
            and its targets as central pixel.
        """
        if self.neighborhood_size is not None:
            central_index = np.floor(self.neighborhood_size / 2).astype(int)
            labels = np.squeeze(samples[:, central_index, central_index])
        else:
            labels = np.squeeze(samples)
        return [samples, labels]


UNMIXING_TRANSFORMS = {
    unmixing_pixel_based_dcae.__name__:
        [ExtractCentralPixelSpectrumTransform,
         SpectralTransform],
    unmixing_cube_based_dcae.__name__:
        [ExtractCentralPixelSpectrumTransform,
         SpectralTransform],

    unmixing_pixel_based_cnn.__name__: [SpectralTransform],
    unmixing_cube_based_cnn.__name__: [SpectralTransform],

    unmixing_rnn_supervised.__name__: [RNNSpectralInputTransform]
}
