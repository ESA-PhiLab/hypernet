import tensorflow as tf


class Transform1D(object):
    def __init__(self, sample_size: int, n_classes: int):
        """
        
        """
        super().__init__()
        self.sample_size = sample_size
        self.n_classes = n_classes

    def __call__(self, sample: tf.Tensor, label: tf.Tensor):
        return [tf.reshape(tf.cast(sample, tf.float32), (self.sample_size, 1)),
                tf.one_hot(tf.cast(label, tf.uint8), (self.n_classes))]
