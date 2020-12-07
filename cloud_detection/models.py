""" Keras models for cloud detection. """

import tensorflow as tf
from tensorflow.keras.layers import (Input, Concatenate, Activation,
                                     Lambda, Conv2D, BatchNormalization,
                                     MaxPool2D, Conv2DTranspose)
from math import floor, ceil


def unet(input_size: int, bn_momentum: float) -> tf.keras.Model:
    """
    Simple U-Net model based on model from
    https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch \
    -for-semantic-segmentation-of-satellite-images-223aa216e705
    consisting of 3 contract blocks and 3 expand blocks.

    :param input_size: Number of input channels, i.e., the number of spectral bands.
    :param bn_momentum: Momentum of the batch normalization layer.
    """
    def contract_block(x: tf.Tensor,
                       filters: int,
                       kernel_size: int,
                       bn_momentum: float) -> tf.Tensor:
        """
        Contracting block of the U-Net.

        :param x: Input to the block.
        :param filters: Number of filters of convolutional layers.
        :param kernel_size: Kernel size of convolutional layers.
        :param bn_momentum: Momentum of the batch normalization layer.
        """
        pad_l, pad_r = ceil(kernel_size/2) - 1, floor(kernel_size/2)
        pad_size = [[0, 0], [pad_l, pad_r], [pad_l, pad_r], [0, 0]]
        x = Lambda(lambda x: tf.pad(x, pad_size, "SYMMETRIC"))(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   padding="valid",
                   activation="relu")(x)
        x = BatchNormalization(momentum=bn_momentum)(x)
        x = Lambda(lambda x: tf.pad(x, pad_size, "SYMMETRIC"))(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   padding="valid",
                   activation="relu")(x)
        x = BatchNormalization(momentum=bn_momentum)(x)
        pool_pad_size = [[0, 0], [1, 1], [1, 1], [0, 0]]
        x = Lambda(lambda x: tf.pad(x, pool_pad_size, "SYMMETRIC"))(x)
        x = MaxPool2D(pool_size=3,
                      strides=2,
                      padding="valid")(x)
        return x

    def expand_block(x: tf.Tensor,
                     filters: int,
                     kernel_size: int,
                     bn_momentum: float) -> tf.Tensor:
        """
        Expanding block of the U-Net.

        :param x: Input to the block.
        :param filters: Number of filters of convolutional layers.
        :param kernel_size: Kernel size of convolutional layers.
        :param bn_momentum: Momentum of the batch normalization layer.
        """
        pad_l, pad_r = ceil(kernel_size/2) - 1, floor(kernel_size/2)
        pad_size = [[0, 0], [pad_l, pad_r], [pad_l, pad_r], [0, 0]]
        x = Lambda(lambda x: tf.pad(x, pad_size, "SYMMETRIC"))(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   padding="valid",
                   activation="relu")(x)
        x = BatchNormalization(momentum=bn_momentum)(x)
        x = Lambda(lambda x: tf.pad(x, pad_size, "SYMMETRIC"))(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   padding="valid",
                   activation="relu")(x)
        x = BatchNormalization(momentum=bn_momentum)(x)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=3,
                            strides=2,
                            padding="same")(x)
        return x

    input_ = Input(shape=(384, 384, input_size))
    concat = Concatenate(axis=-1)

    cont1 = contract_block(input_, 32, 7, bn_momentum)
    cont2 = contract_block(cont1, 64, 3, bn_momentum)
    cont3 = contract_block(cont2, 128, 3, bn_momentum)

    exp1 = expand_block(cont3, 64, 3, bn_momentum)
    exp1cont2 = concat([exp1, cont2])
    exp2 = expand_block(exp1cont2, 32, 3, bn_momentum)
    exp2cont1 = concat([exp2, cont1])
    exp3 = expand_block(exp2cont1, 1, 3, bn_momentum)
    out = Activation(activation="sigmoid")(exp3)

    model = tf.keras.Model(input_, out)

    return model
