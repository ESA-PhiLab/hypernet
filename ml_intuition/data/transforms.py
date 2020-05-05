"""
Module containing all the transformations that can be done on a dataset.
"""

import abc
from typing import List

import numpy as np


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
    def __init__(self):
        """
        Initializer of the spectral transformation.
        """
        super().__init__()

    def __call__(self, sample: np.ndarray, label: np.ndarray) -> List[np.ndarray]:
        """
        Transform 1D samples along the spectral axis.
        Only the spectral features are present for each sample in the dataset.

        :param sample: Input sample that will undergo transformation.
        :param label: Class value for each sample.
        :return: List containing the transformed sample and the class label.
        """
        return [np.expand_dims(sample.astype(np.float), -1), label]


class OneHotEncode(BaseTransform):
    def __init__(self, n_classes: int):
        """
        Initializer of the one-hot encoding transformation.

        :param n_classes: Number of classes.
        """
        super().__init__()
        self.n_classes = n_classes

    def __call__(self, sample: np.ndarray, label: np.ndarray):
        """
        Perform one-hot encoding on incoming label.

        :param sample: Input sample.
        :param label: Class value for each sample that will undergo one-hot encoding.
        :return: List containing the sample and the one-hot encoded class label.
        """
        out_label = np.zeros((label.size, self.n_classes))
        out_label[np.arange(label.size), label] = 1
        return [sample, out_label.astype(np.uint)]


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

    def __call__(self, sample: np.ndarray, label: np.ndarray) -> List[np.ndarray]:
        """"
        Perform min-max normalization on incoming samples.

        :param sample: Input sample that will undergo transformation.
        :param label: Class value for each sample.
        :return: List containing the normalized sample and the class label.
        """
        return [(sample - self.min_) / (self.max_ - self.min_), label]
