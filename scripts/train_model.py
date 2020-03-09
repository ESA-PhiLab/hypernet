"""
Perform the training and validation of the model.
"""

import csv
import json
import os

import clize
import tensorflow as tf
from scripts import metrics, models
from sklearn.metrics import cohen_kappa_score

from ml_intuition.data import io, transforms, utils


def train(*,
          model_name: str,
          kernel_size: int,
          n_kernels: int,
          n_layers: int,
          dest_path: str,
          data_path: str,
          sample_size: int,
          n_classes: int,
          lr: float = 0.005,
          batch_size: int = 150,
          epochs: int = 10,
          verbose: int = 1,
          shuffle: bool = True,
          patience: int = 3):
    """
    Function for training tensorflow models given a dataset.

    :param model_name: Name of the model, it serves as a key in the
        dictionary holding all functions returning models.
    :param kernel_size: Size of ech kernel in each layer.
    :param n_kernels: Number of kernels in each layer.
    :param n_layers: Number of layers in the model.
    :param dest_path: Path to where to save the model under the name "model_name".
    :param data_path: Path to the input data. First dimension of the
        dataset should be the number of samples.
    :param sample_size: Size of the input sample.
    :param n_classes: Number of classes.
    :param lr: Learning rate for the model, i.e., regulates the size of the step
        in the gradient descent process.
    :param batch_size: Size of the batch used in training phase,
        it is the size of samples per gradient step.
    :param epochs: Number of epochs for model to train.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param shuffle: Boolean indicating whether to shuffle dataset
     dataset_key each epoch.
    :param patience: Number of epochs without improvement in order to
        stop the training phase.

    """
    train_dataset, n_train =\
        utils.extract_dataset(batch_size,
                              io.load_data(
                                  data_path, utils.Dataset.TRAIN),
                              [transforms.SpectralTranform(n_classes)])

    val_dataset, n_val =\
        utils.extract_dataset(batch_size,
                              io.load_data(
                                  data_path, utils.Dataset.VAL),
                              [transforms.SpectralTranform(n_classes)])
    if shuffle:
        train_dataset = train_dataset.shuffle(batch_size)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience)

    model = models.get_model(model_key=model_name, kernel_size=kernel_size,
                             n_kernels=n_kernels, n_layers=n_layers,
                             input_size=sample_size, n_classes=n_classes, lr=lr)
    model.summary()
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    time_history = metrics.TimeHistory()
    history = model.fit(x=train_dataset.make_one_shot_iterator(),
                        epochs=epochs,
                        verbose=verbose,
                        shuffle=shuffle,
                        validation_data=val_dataset.make_one_shot_iterator(),
                        callbacks=[callback, time_history],
                        steps_per_epoch=n_train // batch_size,
                        validation_steps=n_val // batch_size)
    model.save(filepath=os.path.join(dest_path, model_name))

    history.history[metrics.TimeHistory.__name__] = time_history.average
    with open(os.path.join(dest_path, 'training_metrics.csv'), 'w') as file:
        write = csv.writer(file)
        write.writerow(history.history.keys())
        write.writerows(zip(*history.history.values()))


if __name__ == '__main__':
    clize.run(train)
