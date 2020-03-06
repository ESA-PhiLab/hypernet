import os

import pytest
import tensorflow as tf
from scripts import train_model

from ml_intuition.data import utils


class TestTrainModels(object):
    @pytest.mark.parametrize(
        'model_path, data_path, batch_size, epochs, '
        + 'verbose, shuffle, patience, sample_size, n_classes',
        [
            # ('model', 'pavia.h5', 15, 5, 2, True, 3, 103, 9),
            ('model', 'pavia.h5', 150, 5, 2, False, 17, 103, 9),
            # ('model', 'pavia.h5', 1, 5, 2, True, 9, 103, 9),
            # ('model', 'pavia.h5', 15, 5, 2, False, 1, 103, 9)
        ])
    def test_training_model(self,
                            model_path,
                            data_path,
                            batch_size,
                            epochs,
                            verbose,
                            shuffle,
                            patience,
                            sample_size,
                            n_classes):
        train_model.train(model_path=model_path,
                          data_path=data_path,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          patience=patience,
                          sample_size=sample_size,
                          n_classes=n_classes)
        assert os.path.exists(utils.Model.TRAINED_MODEL)

    @pytest.mark.parametrize(
        'data_path, batch_size, sample_size, n_classes',
        [
            ('pavia.h5', 15, 103, 9),
        ])
    def test_dataset_extraction(self,
                                data_path: str,
                                batch_size: int,
                                sample_size: int,
                                n_classes: int):
        train_dataset, val_dataset, N_TRAIN, N_VAL =\
            utils.extract_dataset(
                data_path, batch_size, sample_size,
                n_classes, utils.Dataset.TRAIN, utils.Dataset.VAL)
        assert all(isinstance(elem, int) for elem in [N_TRAIN, N_TRAIN])
        assert all(isinstance(elem, tf.data.Dataset)
                   for elem in [train_dataset, val_dataset])
