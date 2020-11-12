"""
All models that are used for training.
"""

import sys
from typing import Union, List

import numpy as np
import tensorflow as tf
from scipy.stats import mode


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
        model.add(
            tf.keras.layers.Conv2D(n_kernels, (kernel_size, 1), strides=(3, 1),
                                   input_shape=(input_size, 1, 1),
                                   padding="valid",
                                   activation='relu'))
        model.add(
            tf.keras.layers.Conv2D(n_kernels, (kernel_size, 1), strides=(2, 1),
                                   input_shape=(input_size, 1, 1),
                                   padding="valid",
                                   activation='relu'))
        model.add(
            tf.keras.layers.Conv2D(n_kernels, (kernel_size, 1), strides=(2, 1),
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


def pool_model_2d(kernel_size: int,
                  n_kernels: int,
                  n_layers: int,
                  input_size: int,
                  n_classes: int) -> tf.keras.Sequential:
    """
    2D model which consists of 2D convolutional layers and 2D pooling layers.

    :param kernel_size: Size of the convolutional kernel.
    :param n_kernels: Number of kernels, i.e., the activation maps in each layer.
    :param n_layers: Number of layers in the network.
    :param input_size: Number of input channels, i.e., the number of spectral bands.
    :param n_classes: Number of classes.
    """

    def add_layer(model):
        model.add(tf.keras.layers.Conv2D(n_kernels, (kernel_size, 1),
                                         strides=(2, 1),
                                         input_shape=(input_size, 1, 1),
                                         padding="valid",
                                         activation='relu'))
        model.add(tf.keras.layers.BatchNormalization()),
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 1), strides=(1, 1)))
        return model

    model = tf.keras.Sequential()

    for _ in range(n_layers):
        model = add_layer(model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
    return model


def get_model(model_key: str, kernel_size: int, n_kernels: int,
              n_layers: int, input_size: int, n_classes: int):
    """
    Get a given instance of model specified by model_key.

    :param model_key: Specifies which model to use.
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


class Ensemble:
    def __init__(self, models: Union[List[tf.keras.Sequential], List[str]],
                 voting: str = 'hard'):
        """
        Ensemble for using multiple models for prediction
        :param models: Either list of tf.keras.models.Sequential models,
            or a list of paths to the models (can't mix both).
        :param voting: If ‘hard’, uses predicted class labels for majority rule
            voting. Else if ‘soft’, predicts the class label based on the argmax
            of the sums of the predicted probabilities.
        """
        if all(type(model) is str for model in models):
            self.models = [tf.keras.models.load_model(model) for model in
                           models]
        elif all(type(model) is tf.keras.models.Sequential for model in models):
            self.models = models
        else:
            raise TypeError("Wrong type of models provided, pass either path "
                            "or the model itself")
        self.voting = voting

    def predict(self, data: Union[np.ndarray, List[np.ndarray]],
                batch_size: int = 1024) -> np.ndarray:
        """
        Return predicted classes
        :param data: Either a single dataset which will be fed into all of the
            models, or a list of datasets, unique for each model
        :param batch_size: Size of the batch used for prediction
        :return: Predicted classes
        """
        predictions = []
        if type(data) is list:
            for model, dataset in zip(self.models, data):
                predictions.append(model.predict(dataset,
                                                 batch_size=batch_size))
            predictions = np.array(predictions)
        else:
            for model in self.models:
                predictions.append(model.predict(data, batch_size=batch_size))
            predictions = np.array(predictions)
        if self.voting == 'hard':
            predictions = np.argmax(predictions, axis=-1)
            return mode(predictions, axis=0)[0][:, 0]
        elif self.voting == 'soft':
            predictions = np.sum(predictions, axis=0)
            predictions = np.argmax(predictions, axis=-1)
        return predictions
