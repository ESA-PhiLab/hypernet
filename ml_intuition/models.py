"""
All models that are used for training.
"""

import sys
import functools
from copy import deepcopy
from typing import Union, List

import numpy as np
import tensorflow as tf
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier


def model_2d(kernel_size: int,
             n_kernels: int,
             n_layers: int,
             input_size: int,
             n_classes: int,
             **kwargs) -> tf.keras.Sequential:
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
                  n_classes: int,
                  **kwargs) -> tf.keras.Sequential:
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


def get_model(model_key: str, **kwargs):
    """
    Get a given instance of model specified by model_key.

    :param model_key: Specifies which model to use.
    :param kwargs: Any keyword arguments that the model accepts.
    """
    # Get the list of all model creating functions and their name as the key:
    all_ = {
        str(f): eval(f) for f in dir(sys.modules[__name__])
    }
    return all_[model_key](**kwargs)


class Ensemble:
    def __init__(self, models: Union[List[tf.keras.Sequential],
                                     List[str],
                                     tf.keras.models.Sequential],
                 voting: str = 'hard'):
        """
        Ensemble for using multiple models for prediction
        :param models: Either list of tf.keras.models.Sequential models,
            or a list of paths to the models (can't mix both).
        :param voting: If ‘hard’, uses predicted class labels for majority rule
            voting. Else if ‘soft’, predicts the class label based on the argmax
            of the sums of the predicted probabilities.
        """
        if type(models) is tf.keras.models.Sequential:
            self.models = models
        elif all(type(model) is str for model in models):
            self.models = [tf.keras.models.load_model(model) for model in
                           models]
        elif all(type(model) is tf.keras.models.Sequential for model in models):
            self.models = models
        else:
            raise TypeError("Wrong type of models provided, pass either path "
                            "or the model itself")
        self.voting = voting
        self.predictor = None

    def generate_models_with_noise(self, copies: int = 5, mean: float = None,
                                   std: float = None, seed=None):
        """
        Generate new models by injecting Gaussian noise into the original
        model's weights.
        :param copies: Number of models to generate
        :param mean: Mean used to draw noise from normal distribution.
            If None, it will be calculated from the layer itself.
        :param std: Standard deviation used to draw noise from normal
            distribution. If None, it will be calculated from the layer itself.
        :param seed: Seed for random number generator.
        :return: None
        """
        assert type(self.models) is tf.keras.models.Sequential, \
            "self.models must be a single model"
        models = [self.models]
        for copy_id in range(copies):
            np.random.seed(seed + copy_id)
            original_weights = deepcopy(self.models.get_weights())
            modified_weights = self.inject_noise_to_weights(original_weights,
                                                            mean, std)
            modified_model = tf.keras.models.clone_model(self.models)
            modified_model.set_weights(modified_weights)
            models.append(modified_model)
        self.models = models

    @staticmethod
    def inject_noise_to_weights(weights: List[np.ndarray], mean: float,
                                std: float):
        """
        Inject noise into all layers
        :param weights: List of weights for each layer
        :param mean: Mean used to draw noise from normal distribution.
        :param std: Std used to draw noise from normal distribution.
        :return: Modified list of weights
        """
        for layer_number, layer_weights in enumerate(weights):
            mean = np.mean(layer_weights) if mean is None else mean
            std = np.std(layer_weights) if std is None else std
            noise = np.random.normal(loc=mean, scale=std * 0.1,
                                     size=layer_weights.shape)
            weights[layer_number] += noise
        return weights

    def _vote(self, voting_method: str, predictions: np.ndarray) -> np.ndarray:
        """
        Perform voting process on provided predictions
        :param voting_method: If ‘hard’, uses predicted class labels for majority rule
            voting. If ‘soft’, predicts the class label based on the argmax
            of the sums of the predicted probabilities. If 'classify', uses a
            classifier which is trained on probabilities
        :param predictions:
        :return:
        """
        pass

    def predict_probabilities(self, data: Union[np.ndarray, List[np.ndarray]],
                              batch_size: int = 1024):
        """
        Return predicted classes
        :param data: Either a single dataset which will be fed into all of the
            models, or a list of datasets, unique for each model
        :param batch_size: Size of the batch used for prediction
        :return: Predicted probabilities for each model and class.
            Shape: [Models, Samples, Classes]
        """
        predictions = []
        if type(data) is list:
            for model, dataset in zip(self.models, data):
                predictions.append(model.predict(dataset,
                                                 batch_size=batch_size))
        else:
            for model in self.models:
                predictions.append(model.predict(data, batch_size=batch_size))
        return np.array(predictions)

    def predict(self, data: Union[np.ndarray, List[np.ndarray]],
                batch_size: int = 1024) -> np.ndarray:
        """
        Return predicted classes
        :param data: Either a single dataset which will be fed into all of the
            models, or a list of datasets, unique for each model
        :param batch_size: Size of the batch used for prediction
        :return: Predicted classes
        """
        predictions = self.predict_probabilities(data, batch_size)
        if self.voting == 'hard':
            predictions = np.argmax(predictions, axis=-1)
            return mode(predictions, axis=0).mode[0, :]
        elif self.voting == 'soft':
            predictions = np.sum(predictions, axis=0)
            predictions = np.argmax(predictions, axis=-1)
        elif self.voting == 'classifier':
            models_count, samples, classes = predictions.shape
            predictions = predictions.swapaxes(0, 1).reshape(samples,
                                                      models_count * classes)
            predictions = self.predictor.predict(predictions)
        return predictions

    def train_ensemble_predictor(self, data: np.ndarray, labels: np.ndarray):
        predictor = RandomForestClassifier()
        models_count, samples, classes = data.shape
        data = data.swapaxes(0, 1).reshape(samples, models_count * classes)
        predictor.fit(data, np.argmax(labels, axis=-1))
        self.predictor = predictor



def model_3d_mfl(kernel_size: int,
                 n_kernels: int,
                 n_classes: int,
                 input_size: int,
                 **kwargs):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=n_kernels,
                                     kernel_size=kernel_size - 3,
                                     strides=(1, 1),
                                     input_shape=(kernel_size, kernel_size,
                                                  input_size),
                                     data_format='channels_last',
                                     padding='valid'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=n_kernels,
                                     kernel_size=(2, 2),
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=n_classes,
                                     kernel_size=(2, 2),
                                     padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Softmax())
    return model


def model_3d_deep(n_classes: int, input_size: int, **kwargs):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(filters=24, kernel_size=3, activation='relu', input_shape=(7, 7, input_size, 1), data_format='channels_last'))
    model.add(tf.keras.layers.Conv3D(filters=24, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Conv3D(filters=24, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
    return model
