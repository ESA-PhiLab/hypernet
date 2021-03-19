import pytest
import tensorflow as tf

from ml_intuition.models import get_model


class TestModels:
    @pytest.mark.parametrize(
        'model_key, kernel_size, n_kernels, n_layers, input_size, n_classes',
        [
            ('model_2d', 4, 3, 1, 103, 9)
        ]
    )
    def test_get_model(self, model_key, kernel_size,
                       n_kernels, n_layers, input_size, n_classes):
        model = get_model(model_key, kernel_size=kernel_size,
                          n_kernels=n_kernels,
                          n_layers=n_layers, input_size=input_size,
                          n_classes=n_classes)
        assert isinstance(model, tf.keras.Sequential), \
            'Assert the model type.'
        layer = model.get_layer(index=2)
        assert isinstance(
            layer, tf.keras.layers.Layer), \
            'Assert not empyt model.'
