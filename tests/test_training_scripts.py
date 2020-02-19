import pytest
import train_models
import os
import utils


class TestTrainModels(object):
    @pytest.mark.parametrize(
        'model_path, data_path, batch_size, epochs, '
        + 'verbose, shuffle, patience, sample_size, n_classes',
        [
            ('model', 'pavia.h5', 15, 5, 2, True, 3, 103, 9)
        ])
    def test_training_model(self, model_path, data_path,
                            batch_size, epochs, verbose,
                            shuffle, patience,
                            sample_size, n_classes):
        assert not os.path.exists(utils.Model.TRAINED_MODEL)
        train_models.train(model_path, data_path, batch_size,
                           epochs, verbose, shuffle, patience,
                           sample_size, n_classes)
        assert os.path.exists(utils.Model.TRAINED_MODEL)
