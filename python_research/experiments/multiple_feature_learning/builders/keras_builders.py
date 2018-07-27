from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import MaxPooling2D, Flatten, Conv2D, Softmax, Input, concatenate
from ..utils.data_types import ModelSettings
from typing import Tuple
import numpy as np


def build_layers(input_shape, kernel_size):
    input1 = Input(shape=input_shape)
    conv2d_1_org = Conv2D(
        filters=200,
        kernel_size=kernel_size,
        strides=(1, 1),
        data_format='channels_last',
        padding='valid'
    )(input1)
    max_pool_2d_1_org = MaxPooling2D(
        pool_size=(2, 2),
        strides=(1, 1),
        padding='same'
    )(conv2d_1_org)
    conv2d_2_org = Conv2D(
        filters=200,
        kernel_size=(2, 2),
        padding='same',
        activation='relu',
        data_format='channels_last'
    )(max_pool_2d_1_org)
    return input1, conv2d_2_org


def build_multiple_features_model(
    settings: ModelSettings,
    no_of_classes: int,
    bands_sets
):
    inputs = []
    conv2ds = []

    for bands in bands_sets:
        if bands is None:
            continue
        layer_input, conv2d = build_layers(
            settings.input_neighbourhood + (bands, ),
            settings.first_conv_kernel_size
        )
        inputs.append(layer_input)
        conv2ds.append(conv2d)

    layers_count = len(conv2ds)
    if layers_count > 1:
        concatenated = concatenate(
            conv2ds,
            axis=2
        )
    else:
        concatenated = conv2ds[0]

    flattened = Flatten()(
        Conv2D(
            filters=no_of_classes,
            kernel_size=(4, 4 * layers_count),
            padding='valid'
        )(concatenated)
    )
    softmax = Softmax()(flattened)
    optimizer = Adam(lr=0.001)
    model = Model(
        inputs=inputs,
        outputs=[softmax]
    )
    model.compile(
        optimizer=optimizer,
        metrics=['accuracy'],
        loss='categorical_crossentropy'
    )

    return model


def build_settings_for_dataset(input_shape: Tuple):
    if not all(np.array(input_shape) >= 5):
        raise ValueError("Input shape has to be greater or equal to 5, was: {}".format(input_shape))
    if all(np.array(input_shape) % 2 == 0):
        raise ValueError("Input shape should have all odd values, had: {}".format(input_shape))
    if not all(np.array(input_shape) == input_shape[0]):
        raise ValueError(
            "All values in the input shape must be equal, were: {}".format(input_shape)
        )
    kernel_size = tuple(np.subtract(input_shape, (3, 3)))
    return ModelSettings(tuple(input_shape), kernel_size)
