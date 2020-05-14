"""
Perform the training of the model.
"""

import os

import clize
import numpy as np
import tensorflow as tf
from clize.parameters import multi

from ml_intuition import enums, models
from ml_intuition.data import io, transforms, utils
from ml_intuition.data.noise import get_noise_functions
from ml_intuition.evaluation import time_metrics


def train(*,
          data,
          model_name: str,
          dest_path: str,
          sample_size: int,
          n_classes: int,
          kernel_size: int = 3,
          n_kernels: int = 16,
          n_layers: int = 1,
          lr: float = 0.005,
          batch_size: int = 150,
          epochs: int = 10,
          verbose: int = 2,
          shuffle: bool = True,
          patience: int = 3,
          seed: int = 0,
          noise: ('post', multi(min=0)),
          noise_sets: ('spost', multi(min=0)),
          noise_params: str = None):
    """
    Function for training tensorflow models given a dataset.

    :param model_name: Name of the model, it serves as a key in the
        dictionary holding all functions returning models.
    :param kernel_size: Size of ech kernel in each layer.
    :param n_kernels: Number of kernels in each layer.
    :param n_layers: Number of layers in the model.
    :param dest_path: Path to where to save the model under the name "model_name".
    :param sample_size: Size of the input sample.
    :param n_classes: Number of classes.
    :param lr: Learning rate for the model, i.e., regulates the size of the step
        in the gradient descent process.
    :param data: Either path to the input data or the data dict itself.
        First dimension of the dataset should be the number of samples.
    :param batch_size: Size of the batch used in training phase,
        it is the size of samples per gradient step.
    :param epochs: Number of epochs for model to train.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param shuffle: Boolean indicating whether to shuffle dataset
     dataset_key each epoch.
    :param patience: Number of epochs without improvement in order to
        stop the training phase.
    :param seed: Seed for training reproducibility.
    :param noise: List containing names of used noise injection methods
        that are performed after the normalization transformations.
    :param noise_sets: List of sets that are affected by the noise injecton methods.
        For this module single element can be either "train" or "val".
    :param noise_params: JSON containing the parameters setting of injection methods.
        Examplary value for this parameter: "{"mean": 0, "std": 1, "pa": 0.1}".
        This JSON should include all parameters for noise injection
        functions that are specified in the noise argument.
        For the accurate description of each parameter, please
        refer to the ml_intuition/data/noise.py module.
    """

    # Reproducibility
    tf.reset_default_graph()
    tf.set_random_seed(seed=seed)
    np.random.seed(seed=seed)

    if type(data) is str:
        train_dict = io.extract_set(data, enums.Dataset.TRAIN)
        val_dict = io.extract_set(data, enums.Dataset.VAL)
        min_, max_ = train_dict[enums.DataStats.MIN], \
            train_dict[enums.DataStats.MAX]
    else:
        train_dict = data[enums.Dataset.TRAIN]
        val_dict = data[enums.Dataset.VAL]
        min_, max_ = data[enums.DataStats.MIN], \
            data[enums.DataStats.MAX]

    transformations = [transforms.SpectralTransform(),
                       transforms.OneHotEncode(n_classes=n_classes),
                       transforms.MinMaxNormalize(min_=min_, max_=max_)]

    tr_transformations = transformations + get_noise_functions(noise, noise_params) \
        if enums.Dataset.TRAIN in noise_sets else transformations
    val_transformations = transformations + get_noise_functions(noise, noise_params) \
        if enums.Dataset.VAL in noise_sets else transformations

    train_dataset, n_train = \
        utils.create_tf_dataset(batch_size,
                                train_dict,
                                tr_transformations)
    val_dataset, n_val = \
        utils.create_tf_dataset(batch_size,
                                val_dict,
                                val_transformations)

    if shuffle:
        train_dataset = train_dataset.shuffle(batch_size)

    model = models.get_model(model_key=model_name, kernel_size=kernel_size,
                             n_kernels=n_kernels, n_layers=n_layers,
                             input_size=sample_size, n_classes=n_classes)
    model.summary()
    model.compile(tf.keras.optimizers.Adam(lr=lr),
                  'categorical_crossentropy',
                  metrics=['accuracy'])

    time_history = time_metrics.TimeHistory()
    mcp_save = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(dest_path, model_name), save_best_only=True,
        monitor='val_acc', mode='max')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience)
    history = model.fit(x=train_dataset.make_one_shot_iterator(),
                        epochs=epochs,
                        verbose=verbose,
                        shuffle=shuffle,
                        validation_data=val_dataset.make_one_shot_iterator(),
                        callbacks=[
                            early_stopping, mcp_save, time_history],
                        steps_per_epoch=n_train // batch_size,
                        validation_steps=n_val // batch_size)

    history.history[time_metrics.TimeHistory.__name__] = time_history.average
    io.save_metrics(dest_path=dest_path,
                    file_name='training_metrics.csv',
                    metrics=history.history)

    np.savetxt(os.path.join(dest_path, 'min-max.csv'),
               np.array([min_, max_]), delimiter=',', fmt='%f')


if __name__ == '__main__':
    clize.run(train)
