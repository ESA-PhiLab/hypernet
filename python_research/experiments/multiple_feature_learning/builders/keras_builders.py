from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import MaxPooling2D, Flatten, Conv2D, Softmax, Input, \
    concatenate
from ..utils.data_types import ModelSettings


def build_layers(input_shape, kernel_size, max_pooling_strides,
                 last_layer_conv_padding):
    input1 = Input(shape=input_shape)
    conv2d_1_org = Conv2D(filters=200,
                          kernel_size=kernel_size,
                          strides=(1, 1),
                          data_format='channels_last',
                          padding='same')(input1)
    max_pool_2d_1_org = MaxPooling2D(pool_size=(2, 2),
                                     strides=max_pooling_strides,
                                     padding='same')(conv2d_1_org)
    conv2d_2_org = Conv2D(filters=200,
                          kernel_size=(2, 2),
                          padding=last_layer_conv_padding,
                          activation='relu',
                          data_format='channels_last')(max_pool_2d_1_org)
    return input1, conv2d_2_org


def build_multiple_features_model(settings: ModelSettings,
                                  no_of_classes: int,
                                  original_bands: int,
                                  area_bands: int,
                                  stddev_bands: int,
                                  diagonal_bands: int,
                                  moment_bands: int):

    input1, conv2d_2_org = build_layers(settings.input_neighbourhood +
                                        (original_bands, ),
                                        settings.first_conv_kernel_size,
                                        settings.max_pooling_strides,
                                        settings.last_layer_conv_padding)
    input2, conv2d_2_area = build_layers(settings.input_neighbourhood +
                                         (area_bands, ),
                                         settings.first_conv_kernel_size,
                                         settings.max_pooling_strides,
                                         settings.last_layer_conv_padding)
    input3, conv2d_2_std = build_layers(settings.input_neighbourhood +
                                        (stddev_bands, ),
                                        settings.first_conv_kernel_size,
                                        settings.max_pooling_strides,
                                        settings.last_layer_conv_padding)
    input4, conv2d_2_diagonal = build_layers(settings.input_neighbourhood +
                                             (diagonal_bands, ),
                                             settings.first_conv_kernel_size,
                                             settings.max_pooling_strides,
                                             settings.last_layer_conv_padding)
    input5, conv2d_2_moment = build_layers(settings.input_neighbourhood +
                                           (moment_bands, ),
                                           settings.first_conv_kernel_size,
                                           settings.max_pooling_strides,
                                           settings.last_layer_conv_padding)

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


def build_single_feature_model(settings: ModelSettings,
                               no_of_classes: int,
                               no_of_bands: int):
    optimizer = Adam(lr=0.001)

    model = Sequential()
    model.add(
        Conv2D(filters=200,
               kernel_size=settings.first_conv_kernel_size,
               strides=(1, 1),
               input_shape=settings.input_neighbourhood + (no_of_bands, ),
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


IndianaSettings = ModelSettings((5, 5), (2, 2), (1, 1), 'valid')
PaviaSettings = ModelSettings((7, 7), (4, 4), (2, 2), 'same')
SalinasSettings = ModelSettings((9, 9), (6, 6), (2, 2), 'valid')


def build_settings_for_dataset(dataset_name: str):
    if dataset_name == 'indiana':
        return IndianaSettings
    if dataset_name == 'pavia':
        return PaviaSettings
    if dataset_name == 'salinas':
        return SalinasSettings
    else:
        raise ValueError("Dataset {} doesn't have predefined settings")
