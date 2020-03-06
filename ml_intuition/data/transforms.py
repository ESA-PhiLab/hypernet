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
        Each subsclass should implement this method.

        :param args: Arbitrary list of arguments.
        :param kwargs: Arbitrary dictionary of arguments.
        """
        pass


class SpectralTranform(BaseTransform):
    def __init__(self, sample_size: int, n_classes: int):
        """
        Initializer of the transorm class.

        :param sample_size: Spectral size of the sample.
        :param n_classes: Number of classes.
        """
        super().__init__()
        self.sample_size = sample_size
        self.n_classes = n_classes

    def __call__(self, sample: tf.Tensor, label: tf.Tensor) -> List[tf.Tensor]:
        """
        Transform 1D samples along the spectral axis.
        Only the spectral features are present for each sample in the dataset.        

        :param sample: Input sample that will undergo transformation.
        :param label: Class value for each sample that will undergo one-hot-encoding.
        :return: List containing the transformed sample and the class label.
        """
        return [tf.reshape(tf.cast(sample, tf.float32), (self.sample_size, 1)),
                tf.one_hot(tf.cast(label, tf.uint8), self.n_classes)]
