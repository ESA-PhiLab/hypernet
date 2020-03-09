import sys

import tensorflow as tf


def model_2d(kernel_size: int,
             n_kernels: int,
             n_layers: int,
             input_size: tuple,
             n_classes: int,
             lr: float = 0.001):
    """
    2D model which consists of 2D convolutional blocks, max pooling and batch normalization.

    :param kernel_size: Size of the convolutional kernel.
    :param n_kernels: Number of kernels, i.e., the activation maps in each layer.
    :param n_layers: Number of layers in the network.
    :param input_size: Number of input channels, i.e., the number of spectral bands.
    :param n_classes: Number of classes.
    :param ls: Learning rate, it regulates the gradient size in the optimization step.
    """
    def add_layer(model):
        model.add(tf.keras.layers.Conv2D(n_kernels, (kernel_size, 1),
                                         input_shape=(input_size, 1, 1),
                                         padding="valid",
                                         activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 1)))
        return model

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(n_kernels, (kernel_size, 1),
                                     input_shape=(input_size, 1, 1),
                                     padding="valid",
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 1)))
    if n_layers > 1:
        for _ in range(n_layers - 1):
            model = add_layer(model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=200, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
    return model


def get_model(model_key: str, kernel_size: int, n_kernels: int,
              n_layers: int, input_size: int, n_classes: int,
              lr: float = 0.001):
    all_ = {
        str(f): eval(f) for f in dir(sys.modules[__name__])
    }
    return all_[model_key](kernel_size=kernel_size,
                           n_kernels=n_kernels, n_layers=n_layers,
                           input_size=input_size, n_classes=n_classes, lr=lr)
