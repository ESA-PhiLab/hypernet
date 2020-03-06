"""
All transformations that can be done on a dataset.
"""

import abc

import tensorflow as tf


class BaseTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class SpectralTranform(BaseTransform):
    def __init__(self, sample_size: int, n_classes: int):
        """
        Transform 1D samples along the spectral axis.
        Only the spectral features are present in the dataset.
        """
        super().__init__()
        self.sample_size = sample_size
        self.n_classes = n_classes

    def __call__(self, sample: tf.Tensor, label: tf.Tensor) -> list:
        return [tf.reshape(tf.cast(sample, tf.float32), (self.sample_size, 1)),
                tf.one_hot(tf.cast(label, tf.uint8), self.n_classes)]
