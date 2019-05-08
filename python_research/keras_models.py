from typing import Tuple, NamedTuple
import numpy as np
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import MaxPooling2D, Flatten, Conv2D, Softmax, Input, \
    concatenate, Conv1D, MaxPooling1D, Dense, BatchNormalization


class ModelSettings(NamedTuple):
    input_neighbourhood: Tuple[int, int]
    first_conv_kernel_size: Tuple[int, int]


def build_layers(input_shape, kernel_size):
    input1 = Input(shape=input_shape)
    conv2d_1_org = Conv2D(filters=200,
                          kernel_size=kernel_size,
                          strides=(1, 1),
                          data_format='channels_last',
                          padding='valid')(input1)
    max_pool_2d_1_org = MaxPooling2D(pool_size=(2, 2),
                                     strides=(1, 1),
                                     padding='same')(conv2d_1_org)
    conv2d_2_org = Conv2D(filters=200,
                          kernel_size=(2, 2),
                          padding='same',
                          activation='relu',
                          data_format='channels_last')(max_pool_2d_1_org)
    return input1, conv2d_2_org


def build_multiple_features_model(settings: ModelSettings,
                                  no_of_classes: int,
                                  bands_set):

    input1, conv2d_2_org = build_layers(settings.input_neighborhood +
                                        (bands_set[0], ),
                                        settings.first_conv_kernel_size)
    input2, conv2d_2_area = build_layers(settings.input_neighborhood +
                                         (bands_set[1], ),
                                         settings.first_conv_kernel_size)
    input3, conv2d_2_std = build_layers(settings.input_neighborhood +
                                        (bands_set[2], ),
                                        settings.first_conv_kernel_size)
    input4, conv2d_2_diagonal = build_layers(settings.input_neighborhood +
                                             (bands_set[3], ),
                                             settings.first_conv_kernel_size)
    input5, conv2d_2_moment = build_layers(settings.input_neighborhood +
                                           (bands_set[4], ),
                                           settings.first_conv_kernel_size)

    concatenated = concatenate(
        [conv2d_2_org,
         conv2d_2_area,
         conv2d_2_std,
         conv2d_2_diagonal,
         conv2d_2_moment], axis=2)
    conv2d = Conv2D(filters=no_of_classes,
                    kernel_size=(4, 20),
                    padding='valid')(concatenated)
    flattened = Flatten()(conv2d)
    softmax = Softmax()(flattened)
    model = Model(inputs=[input1,
                          input2,
                          input3,
                          input4,
                          input5],
                  outputs=[softmax])
    model.compile(optimizer='adam',
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')
    return model


def build_3d_model(settings: ModelSettings,
                   no_of_classes: int,
                   no_of_bands: int):
    optimizer = Adam(lr=0.001)

    model = Sequential()
    model.add(
        Conv2D(filters=200,
               kernel_size=settings.first_conv_kernel_size,
               strides=(1, 1),
               input_shape=settings.input_neighborhood + (no_of_bands, ),
               data_format='channels_last',
               padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           padding='valid'))
    model.add(Conv2D(filters=200,
                     kernel_size=(2, 2),
                     padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=no_of_classes,
                     kernel_size=(2, 2),
                     padding='valid'))
    model.add(Flatten())
    model.add(Softmax())
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')
    return model


def build_1d_model(input_shape, filters, kernel_size, classes_count, blocks=1):
    optimizer = Adam(lr=0.0001)

    def add_block(model):
        model.add(
            Conv1D(filters, kernel_size, padding="valid", activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        return model

    model = Sequential()
    model.add(Conv1D(filters, kernel_size, input_shape=input_shape, padding="valid", activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    if blocks > 1:
        for block in range(blocks - 1):
            model = add_block(model)
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=classes_count, activation='softmax'))
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')
    return model


def build_1d_model_2(input_shape, kernel_size, classes_count, blocks=1):

    def add_second_block(model):
        model.add(Conv1D(filters=10, kernel_size=30, activation='relu'))
        model.add(BatchNormalization())
        return model

    def add_third_block(model):
        model.add(Conv1D(filters=10, kernel_size=10, activation='relu'))
        model.add(BatchNormalization())
        return model

    optimizer = Adam(lr=0.0001)

    model = Sequential()
    model.add(Conv1D(kernel_size=kernel_size, filters=20, input_shape=input_shape, padding="valid", activation='relu'))
    model.add(BatchNormalization())
    if blocks > 1:
        model = add_second_block(model)
    if blocks > 2:
        model = add_third_block(model)

    model.add(Flatten())
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(units=classes_count, activation='softmax'))
    model.compile(optimizer=optimizer, metrics=['accuracy'],
                  loss='categorical_crossentropy')
    return model


def build_settings_for_dataset(input_shape: Tuple):
    if not all(np.array(input_shape) >= 5):
        raise ValueError("Input shape has to be greater or equal to 5, was: {}".format(input_shape))
    if all(np.array(input_shape) % 2 == 0):
        raise ValueError("Input shape should have all odd values, had: {}".format(input_shape))
    if not all(np.array(input_shape) == input_shape[0]):
        raise ValueError("All values in the input shape must be equal, were: {}".format(input_shape))
    kernel_size = tuple(np.subtract(input_shape, (3, 3)))
    return ModelSettings(tuple(input_shape), kernel_size)
