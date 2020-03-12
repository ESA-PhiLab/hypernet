"""
Module containing all the transformations that can be done on a dataset.
"""

import abc
from typing import List

import tensorflow as tf


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

    def __call__(self, sample: tf.Tensor, label: tf.Tensor) -> List[tf.Tensor]:
        """
        Transform 1D samples along the spectral axis.
        Only the spectral features are present for each sample in the dataset.

        :param sample: Input sample that will undergo transformation.
        :param label: Class value for each sample.
        :return: List containing the transformed sample and the class label.
        """
        return [tf.expand_dims(tf.cast(sample, tf.float32), -1), label]


class OneHotEncode(BaseTransform):
    def __init__(self, n_classes: int):
        """
        Initializer of the one-hot encoding transformation.

        :param n_classes: Number of classes.
        """
        super().__init__()
        self.n_classes = n_classes

    def __call__(self, sample: tf.Tensor, label: tf.Tensor):
        """
        Perform one-hot encoding on incoming label.

        :param sample: Input sample.
        :param label: Class value for each sample that will undergo one-hot encoding.
        :return: List containing the sample and the one-hot encoded class label.
        """
        return [sample, tf.one_hot(tf.cast(label, tf.uint8), self.n_classes)]


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

    def __call__(self, sample: tf.Tensor, label: tf.Tensor) -> List[tf.Tensor]:
        """"
        Perform min-max normalization on incoming samples.

        :param sample: Input sample that will undergo transformation.
        :param label: Class value for each sample.
        :return: List containing the normalized sample and the class label.
        """
        return [(sample - self.min_) / (self.max_ - self.min_), label]
