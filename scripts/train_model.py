"""
Perform the training of the model.
"""

import os

import clize
import numpy as np
import tensorflow as tf
from clize.parameters import multi

from ml_intuition import enums, models
from ml_intuition.data import io, transforms
from ml_intuition.data.noise import get_noise_functions
from ml_intuition.data.preprocessing import reshape_cube_to_2d_samples, reshape_cube_to_3d_samples
from ml_intuition.data.utils import get_central_pixel_spectrum
from ml_intuition.evaluation import time_metrics, performance_metrics
from ml_intuition.models import check_for_autoencoder


def train(*,
          data,
          model_name: str,
          dest_path: str,
          sample_size: int,
          n_classes: int,
          neighborhood_size: int = None,
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
          use_unmixing: bool = False,
          endmembers_path: str = None,
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
    :param neighborhood_size: Size of the spatial patch.
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
    :param use_unmixing: Boolean indicating whether to perform experiments on the unmixing datasets,
        where classes in each pixel are present as abundances fractions.
    :param endmembers_path: Path to the endmembers file containing average reflectances for each class.
        Used only when use_unmixing is true.
    :param noise: List containing names of used noise injection methods
        that are performed after the normalization transformations.
    :param noise_sets: List of sets that are affected by the noise injection methods.
        For this module single element can be either "train" or "val".
    :param noise_params: JSON containing the parameters setting of injection methods.
        Exemplary value for this parameter: "{"mean": 0, "std": 1, "pa": 0.1}".
        This JSON should include all parameters for noise injection
        functions that are specified in the noise argument.
        For the accurate description of each parameter, please
        refer to the ml_intuition/data/noise.py module.
    """
    # Reproducibility:
    tf.reset_default_graph()
    tf.set_random_seed(seed=seed)
    np.random.seed(seed=seed)

    is_autoencoder = check_for_autoencoder(model_name)
    # Set the training specifications:
    model_kwargs = {'kernel_size': kernel_size,
                    'n_kernels': n_kernels,
                    'n_layers': n_layers,
                    'input_size': sample_size,
                    'n_classes': n_classes,
                    'neighborhood_size': neighborhood_size,
                    'endmembers': np.load(endmembers_path) if endmembers_path is not None else None}
    model = models.get_model(model_key=model_name, **model_kwargs)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss=performance_metrics.get_loss(model_name, use_unmixing),
                  metrics=list(
                      performance_metrics.get_unmixing_metrics(model_name, use_unmixing, 'TRAIN').values())
                  if use_unmixing else ['accuracy'])

    time_history = time_metrics.TimeHistory()
    monitor = performance_metrics.get_checkpoint_monitor_quantity(use_unmixing, is_autoencoder)
    mcp_save = tf.keras.callbacks.ModelCheckpoint(os.path.join(dest_path, model_name),
                                                  save_best_only=True,
                                                  monitor=monitor,
                                                  mode='min' if use_unmixing else 'max')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss' if is_autoencoder else 'val_loss',
                                                      patience=patience, mode='min')
    callbacks = [time_history, mcp_save, early_stopping]

    if is_autoencoder:
        data = transforms.apply_transformations(data, [transforms.PerBandMinMaxNormalization(
            **transforms.PerBandMinMaxNormalization.get_min_max_vectors((data['data'])))])
        if neighborhood_size is None:
            data['data'], data['labels'] = reshape_cube_to_2d_samples(data['data'], data['labels'], -1, use_unmixing)

        else:
            data['data'], data['labels'] = reshape_cube_to_3d_samples(data['data'], data['labels'],
                                                                      neighborhood_size, -1, -1, use_unmixing)
        data['data'] = np.expand_dims(data['data'], -1)
        history = model.fit(x=data['data'], y=get_central_pixel_spectrum(data['data'], neighborhood_size),
                            epochs=epochs, verbose=verbose,
                            shuffle=shuffle, callbacks=callbacks, batch_size=batch_size)
        history.history[time_metrics.TimeHistory.__name__] = time_history.average
        io.save_metrics(dest_path=dest_path, file_name='training_metrics.csv', metrics=history.history)
        return

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

    transformations = [transforms.SpectralTransform(), transforms.MinMaxNormalize(min_=min_, max_=max_)]
    if not use_unmixing:
        transformations += transforms.OneHotEncode(n_classes=n_classes)

    tr_transformations = transformations + get_noise_functions(noise, noise_params) \
        if enums.Dataset.TRAIN in noise_sets else transformations
    val_transformations = transformations + get_noise_functions(noise, noise_params) \
        if enums.Dataset.VAL in noise_sets else transformations

    train_dict = transforms.apply_transformations(train_dict, tr_transformations)
    val_dict = transforms.apply_transformations(val_dict, val_transformations)

    history = model.fit(x=train_dict[enums.Dataset.DATA], y=train_dict[enums.Dataset.LABELS],
                        epochs=epochs, verbose=verbose, shuffle=shuffle,
                        validation_data=(val_dict[enums.Dataset.DATA], val_dict[enums.Dataset.LABELS]),
                        callbacks=callbacks, batch_size=batch_size)

    np.savetxt(os.path.join(dest_path, 'min-max.csv'), np.array([min_, max_]), delimiter=',', fmt='%f')
    history.history[time_metrics.TimeHistory.__name__] = time_history.average
    io.save_metrics(dest_path=dest_path, file_name='training_metrics.csv', metrics=history.history)


if __name__ == '__main__':
    clize.run(train)
