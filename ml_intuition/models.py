"""
All models that are used for training.
"""
import json
import sys
from copy import deepcopy
from typing import Union, List

import numpy as np
import tensorflow as tf
import yaml
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def model_2d(kernel_size: int,
             n_kernels: int,
             n_layers: int,
             input_size: int,
             n_classes: int,
             **kwargs) -> tf.keras.Sequential:
    """
    2D model which consists of 2D convolutional blocks.

    :param kernel_size: Size of the convolutional kernel.
    :param n_kernels: Number of kernels, i.e.,
        the activation maps in each layer.
    :param n_layers: Number of layers in the network.
    :param input_size: Number of input channels, i.e.,
        the number of spectral bands.
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
    :param n_kernels: Number of kernels, i.e.,
        the activation maps in each layer.
    :param n_layers: Number of layers in the network.
    :param input_size: Number of input channels, i.e.,
        the number of spectral bands.
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
    model.add(
        tf.keras.layers.Conv3D(filters=24, kernel_size=3, activation='relu',
                               input_shape=(7, 7, input_size, 1),
                               data_format='channels_last'))
    model.add(
        tf.keras.layers.Conv3D(filters=24, kernel_size=3, activation='relu'))
    model.add(
        tf.keras.layers.Conv3D(filters=24, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
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
    MODELS = {
        'RFR': RandomForestRegressor,
        'RFC': RandomForestClassifier,
        'SVR': SVR,
        'SVC': SVC,
        'DTR': DecisionTreeRegressor,
        'DTC': DecisionTreeClassifier,
        None: DecisionTreeClassifier
    }

    def __init__(self, models: Union[List[tf.keras.Sequential],
                                     List[str],
                                     tf.keras.models.Sequential,
                                     None] = None,
                 voting: str = 'hard'):
        """
        Ensemble for using multiple models for prediction
        :param models: Either list of tf.keras.models.Sequential models,
            or a list of paths to the models (can't mix both).
        :param voting: If ‘hard’, uses predicted class labels for majority rule
            voting. Else if ‘soft’, predicts the class label based on the argmax
            of the sums of the predicted probabilities.
        """
        self.models = models
        self.voting = voting
        self.predictor = None

    def generate_models_with_noise(self, copies: int = 5, mean: float = None,
                                   std: float = None, seed=None) -> None:
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
                                std: float) -> List[np.ndarray]:
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

    def vote(self, predictions: np.ndarray) -> np.ndarray:
        """
        Perform voting process on provided predictions
        :param predictions: Predictions of all models as a numpy array.
        :return: Predicted classes.
        """
        if self.voting == 'hard':
            predictions = np.argmax(predictions, axis=-1)
            return mode(predictions, axis=0).mode[0, :]
        elif self.voting == 'soft':
            predictions = np.sum(predictions, axis=0)
            return np.argmax(predictions, axis=-1)
        elif self.voting == 'booster':
            models_count, samples, classes = predictions.shape
            predictions = predictions.swapaxes(0, 1).reshape(
                samples, models_count * classes)
            return self.predictor.predict(predictions)
        elif self.voting == 'mean':
            return np.asarray(predictions).mean(axis=0)

    def predict_probabilities(self, data: Union[np.ndarray, List[np.ndarray]],
                              batch_size: int = 1024) -> np.ndarray:
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
        return self.vote(predictions)

    def train_ensemble_predictor(self, data: np.ndarray,
                                 labels: np.ndarray,
                                 predictor: str = None,
                                 model_params: str = None):
        try:
            model_params = json.loads(model_params)
        except json.decoder.JSONDecodeError:
            model_params = yaml.load(model_params)
        model = self.MODELS[predictor](**model_params)
        if predictor == 'SVR':
            # If the model is an SVR, extend its functionality
            # to multi-target regression:
            model = MultiOutputRegressor(model)
        models_count, samples, classes = data.shape
        data = data.swapaxes(0, 1).reshape(samples, models_count * classes)
        self.predictor = model.fit(data, labels)


def unmixing_pixel_based_cnn(n_classes: int, input_size: int,
                             **kwargs) -> tf.keras.Sequential:
    """
    Model for supervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):

    Zhang, Xiangrong, Yujia Sun, Jingyan Zhang, Peng Wu, and Licheng Jiao.
    "Hyperspectral unmixing via deep convolutional neural networks."
    IEEE Geoscience and Remote Sensing Letters 15, no. 11 (2018): 1755-1759.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv3D(filters=3, kernel_size=(1, 1, 5),
                               activation='relu',
                               input_shape=(1, 1, input_size, 1),
                               data_format='channels_last'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=6, kernel_size=(1, 1, 4),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=12, kernel_size=(1, 1, 5),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=24, kernel_size=(1, 1, 4),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=192, activation='relu'))
    model.add(tf.keras.layers.Dense(units=150, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
    return model


def unmixing_cube_based_cnn(n_classes: int, input_size: int,
                            **kwargs) -> tf.keras.Sequential:
    """
    Model for supervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):

    Zhang, Xiangrong, Yujia Sun, Jingyan Zhang, Peng Wu, and Licheng Jiao.
    "Hyperspectral unmixing via deep convolutional neural networks."
    IEEE Geoscience and Remote Sensing Letters 15, no. 11 (2018): 1755-1759.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(filters=16, kernel_size=(1, 1, 5),
                                     activation='relu',
                                     input_shape=(kwargs['neighborhood_size'],
                                                  kwargs['neighborhood_size'],
                                                  input_size, 1),
                                     data_format='channels_last'))
    model.add(tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 4),
                                     activation='relu'))
    model.add(tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 5),
                                     activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Conv3D(filters=128, kernel_size=(1, 1, 4),
                                     activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=192, activation='relu'))
    model.add(tf.keras.layers.Dense(units=150, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
    return model


def unmixing_pixel_based_dcae(n_classes: int, input_size: int,
                              **kwargs) -> tf.keras.Sequential:
    """
    Model for unsupervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):

    Khajehrayeni, Farshid, and Hassan Ghassemian.
    "Hyperspectral unmixing using deep convolutional autoencoders
    in a supervised scenario."
    IEEE Journal of Selected Topics in Applied Earth Observations
    and Remote Sensing 13 (2020): 567-576.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(filters=2, kernel_size=(1, 1, 3),
                                     activation='relu',
                                     input_shape=(1, 1, input_size, 1),
                                     data_format='channels_last'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=4, kernel_size=(1, 1, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=8, kernel_size=(1, 1, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=16, kernel_size=(1, 1, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='relu'))
    model.add(tf.keras.layers.Softmax())
    # Decoder part (later to be dropped):
    model.add(tf.keras.layers.Dense(units=input_size, activation='relu'))
    # Set the endmembers weights to be equal to the endmembers matrix i.e.,
    # the spectral signatures of each class:
    model.layers[-1].set_weights(
        (np.swapaxes(kwargs['endmembers'], 1, 0), np.zeros(input_size)))
    # Freeze the last layer which must be equal to endmembers
    # and residual term (zero vector):
    model.layers[-1].trainable = False
    return model


def unmixing_cube_based_dcae(n_classes: int, input_size: int,
                             **kwargs) -> tf.keras.Sequential:
    """
    Model for unsupervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):

    Khajehrayeni, Farshid, and Hassan Ghassemian.
    "Hyperspectral unmixing using deep convolutional autoencoders
    in a supervised scenario."
    IEEE Journal of Selected Topics in Applied Earth Observations
    and Remote Sensing 13 (2020): 567-576.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 3),
                                     activation='relu',
                                     input_shape=(kwargs['neighborhood_size'],
                                                  kwargs['neighborhood_size'],
                                                  input_size, 1),
                                     data_format='channels_last'))
    model.add(tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.Conv3D(filters=128, kernel_size=(1, 1, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='relu'))
    model.add(tf.keras.layers.Softmax())
    # Decoder part (later to be dropped):
    model.add(tf.keras.layers.Dense(units=input_size, activation='linear'))
    # Set the endmembers weights to be equal to the endmembers matrix i.e.,
    # the spectral signatures of each class:
    model.layers[-1].set_weights(
        (np.swapaxes(kwargs['endmembers'], 1, 0), np.zeros(input_size)))
    # Freeze the last layer which must be equal to endmembers
    # and residual term (zero vector):
    model.layers[-1].trainable = False
    return model


def unmixing_rnn_supervised(n_classes: int, **kwargs) -> tf.keras.Sequential:
    """
    Model for the unmixing which utilizes a recurrent neural network (RNN)
    for extracting valuable information from the spectral domain
    in an supervised manner.

    :param n_classes: Number of classes.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.GRU(units=8, input_shape=(kwargs['input_size'], 1),
                            return_sequences=True))
    model.add(tf.keras.layers.GRU(units=32, return_sequences=True))
    model.add(tf.keras.layers.GRU(units=128, return_sequences=True))
    model.add(tf.keras.layers.GRU(units=512, return_sequences=False))

    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    return model


ML_MODELS = {
    'decision_tree_clf': DecisionTreeClassifier,
    'random_forest_clf': RandomForestClassifier,
    'decision_tree_reg': DecisionTreeRegressor,
    'random_forest_reg': RandomForestRegressor,
}

ML_MODELS_GRID = {
    'decision_tree_clf':
        {
            'max_depth': [2, 4, 6, 8, 10, 20, 50, 100],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 4, 6, 8, 10, 20, 50]
        },

    'random_forest_clf':
        {
            'max_depth': [10, 20, 40, 60],
            'min_samples_leaf': [1, 2, 4],
            'n_estimators': [200, 400, 600, 800]
        },
    'decision_tree_reg':
        {
            'max_depth': [2, 4, 6, 8, 10, 20, 50, 100],
            'min_samples_split': [2, 4, 6, 8, 10, 20, 50]
        },
    'random_forest_reg':
        {
            'max_depth': [10, 20, 40, 60],
            'min_samples_leaf': [1, 2, 4],
            'n_estimators': [200, 400, 600, 800]
        },
}
