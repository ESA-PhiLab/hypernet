import os

import pytest

import train_models
import utils


class TestTrainModels(object):
    @pytest.mark.parametrize(
        'model_path, data_path, batch_size, epochs, '
        + 'verbose, shuffle, patience, sample_size, n_classes',
        [
            ('model', 'pavia.h5', 15, 5, 2, True, 3, 103, 9),
            ('model', 'pavia.h5', 150, 5, 2, False, 17, 103, 9),
            ('model', 'pavia.h5', 1, 5, 2, True, 9, 103, 9),
            ('model', 'pavia.h5', 15, 5, 2, False, 1, 103, 9)


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
        train_models.train(model_path=model_path,
                           data_path=data_path,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=verbose,
                           shuffle=shuffle,
                           patience=patience,
                           sample_size=sample_size,
                           n_classes=n_classes)
        assert os.path.exists(utils.Model.TRAINED_MODEL)

    def test_dataset_extraction(self,
                                data_path: str,
                                batch_size: int,
                                sample_size: int,
                                n_classes: int):
        train_dataset, val_dataset, N_TRAIN, N_VAL =\
            train_models._extract_trainable_datasets(
                data_path, batch_size, sample_size, n_classes)
        assert all(isinstance(elem, int) for elem in [N_TRAIN, N_TRAIN])
