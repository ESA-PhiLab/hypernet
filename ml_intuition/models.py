"""
All models that are used for training.
"""

import sys

import tensorflow as tf


def model_2d(kernel_size: int,
             n_kernels: int,
             n_layers: int,
             input_size: int,
             n_classes: int) -> tf.keras.Sequential:
    """
    2D model which consists of 2D convolutional blocks.

    :param kernel_size: Size of the convolutional kernel.
    :param n_kernels: Number of kernels, i.e., the activation maps in each layer.
    :param n_layers: Number of layers in the network.
    :param input_size: Number of input channels, i.e., the number of spectral bands.
    :param n_classes: Number of classes.
    """
    def add_layer(model):
        model.add(tf.keras.layers.Conv2D(n_kernels, (kernel_size, 1),
                                         input_shape=(input_size, 1, 1),
                                         padding="valid",
                                         activation='relu'))
        model.add(tf.keras.layers.Conv2D(n_kernels, (kernel_size, 1), strides=(3, 1),
                                         input_shape=(input_size, 1, 1),
                                         padding="valid",
                                         activation='relu'))
        model.add(tf.keras.layers.Conv2D(n_kernels, (kernel_size, 1), strides=(2, 1),
                                         input_shape=(input_size, 1, 1),
                                         padding="valid",
                                         activation='relu'))
        model.add(tf.keras.layers.Conv2D(n_kernels, (kernel_size, 1), strides=(2, 1),
                                         input_shape=(input_size, 1, 1),
                                         padding="valid",
                                         activation='relu'))
        return model
    model = tf.keras.Sequential()

    for _ in range(n_layers):
        model = add_layer(model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=200, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
    return model


def get_model(model_key: str, kernel_size: int, n_kernels: int,
              n_layers: int, input_size: int, n_classes: int):
    """
    Get a given instance of model specified by mode_key.

    :param model_key: Specifes which model to use.
    :param kernel_size: Size of the convolutional kernel.
    :param n_kernels: Number of kernels, i.e., the activation maps in each layer.
    :param n_layers: Number of layers in the network.
    :param input_size: Number of input channels, i.e., the number of spectral bands.
    :param n_classes: Number of classes.
    """
    # Get the list of all model creating functions and their name as the key:
    all_ = {
        str(f): eval(f) for f in dir(sys.modules[__name__])
    }
    return all_[model_key](kernel_size=kernel_size,
                           n_kernels=n_kernels, n_layers=n_layers,
                           input_size=input_size, n_classes=n_classes)
