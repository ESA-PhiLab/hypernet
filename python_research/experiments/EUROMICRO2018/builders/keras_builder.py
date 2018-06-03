import numpy as np
from typing import Tuple, List

import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Conv2D, Dense, Flatten, MaxPooling1D, Reshape, Dropout, MaxPooling2D
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.regularizers import l2
from keras.initializers import he_normal


def build_conv1(input_shape: Tuple, num_classes: int):
    inputs = Input(shape=input_shape, name='input0')
    conv1 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs)
    conv2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    flatten = Flatten()(conv2)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    outputs = Dense(units=num_classes, activation='sigmoid', name='output0')(dense2)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(0.001), loss=categorical_crossentropy, metrics=['accuracy'])

    return model


def build_dense1(input_shape: Tuple, num_classes: int, hidden_size: int=80, seed: int=0):
    inputs = Input(shape=input_shape, name='input0')
    flatten = Flatten()(inputs)
    dense1 = Dense(units=hidden_size, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=he_normal(seed))(flatten)
    dense2 = Dense(units=hidden_size, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=he_normal(seed))(dense1)
    dense3 = Dense(units=hidden_size, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=he_normal(seed))(dense2)
    dense4 = Dense(units=hidden_size, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=he_normal(seed))(dense3)
    outputs = Dense(units=num_classes, activation='sigmoid',
                    kernel_regularizer=l2(0.01), kernel_initializer=he_normal(seed), name='output0')(dense4)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(0.0001), loss=categorical_crossentropy, metrics=['accuracy'])

    return model


def build_conv1d(input_shape: Tuple, num_classes: int):
    inputs = Input(shape=input_shape, name='input0')
    conv1 = Conv1D(filters=20, kernel_size=(25,), strides=(1,), padding='valid', activation='relu')(inputs)
    maxpool = MaxPooling1D(pool_size=(5,), strides=(5,), padding='valid')(conv1)
    flatten = Flatten()(maxpool)
    # dense1 = Dense(units=100, activation='sigmoid')(flatten)
    dense2 = Dense(units=num_classes, activation='sigmoid')(flatten)
    # dense2 = Dense(units=128, activation='relu')(dense1)
    # outputs = Dense(units=num_classes, activation='sigmoid', name='output0')(dense2)

    model = Model(inputs=[inputs], outputs=[dense2])
    model.compile(optimizer=Adam(0.001), loss=categorical_crossentropy, metrics=['accuracy'])

    return model


def build_restore_dense(input_shape: Tuple, hidden_size: int, num_classes: int, seed: int=0):
    inputs = Input(shape=input_shape, name='input0')
    flatten = Flatten()(inputs)
    dense1 = Dense(units=hidden_size, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=he_normal(seed))(flatten)
    dense2 = Dense(units=hidden_size, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=he_normal(seed))(dense1)
    dense3 = Dense(units=hidden_size, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=he_normal(seed))(dense2)
    dense4 = Dense(units=hidden_size, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=he_normal(seed))(dense3)
    dense5 = Dense(units=input_shape[0], activation='relu',
                    kernel_regularizer=l2(0.01), kernel_initializer=he_normal(seed), name='output0')(dense4)
    outputs = Reshape(input_shape)(dense5)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(0.005, clipnorm=1.0), loss=mean_squared_error, metrics=['accuracy'])

    return model


def hu(input_shape: Tuple, num_classes: int):
    inputs = Input(shape=input_shape, name='input0')
    conv1 = Conv1D(filters=93, kernel_size=(11,), strides=(1,), padding='valid', activation='relu')(inputs)
    maxpool = MaxPooling1D(pool_size=(3,), strides=(3,), padding='valid')(conv1)
    flatten = Flatten()(maxpool)
    # dropout = Dropout(0.2)(flatten)
    dense1 = Dense(units=100, activation='relu')(flatten)
    dense2 = Dense(units=num_classes, activation='softmax', name='output0')(dense1)
    # dense2 = Dense(units=128, activation='relu')(dense1)
    # outputs = Dense(units=num_classes, activation='sigmoid', name='output0')(dense2)

    model = Model(inputs=[inputs], outputs=[dense2])
    model.compile(optimizer=Adam(0.001), loss=categorical_crossentropy, metrics=['accuracy'])

    return model


def uff_hu(input_shape: Tuple, num_classes: int):
    inputs = Input(shape=input_shape, name='input0')
    conv1 = Conv2D(filters=100, kernel_size=(1, 11), strides=(1,1),
                   padding='valid', activation='relu', name='conv_layer_1')(inputs)
    maxpool = MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='valid')(conv1)
    flatten = Flatten()(maxpool)
    # dropout = Dropout(0.2)(flatten)
    dense1 = Dense(units=100, activation='relu')(flatten)
    dense2 = Dense(units=num_classes, activation='softmax', name='output')(dense1)
    # dense2 = Dense(units=128, activation='relu')(dense1)
    # outputs = Dense(units=num_classes, activation='sigmoid', name='output0')(dense2)

    model = Model(inputs=[inputs], outputs=[dense2])
    model.compile(optimizer=Adam(0.001), loss=categorical_crossentropy, metrics=['accuracy'])

    return model


def full_conv(input_shape: Tuple, num_classes: int, filters: Tuple=(40, 60, 80)):
    inputs = Input(shape=input_shape, name='input0')

    conv1 = Conv2D(filters=filters[0], kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(inputs)
    mp1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(conv1)

    conv2 = Conv2D(filters=filters[1], kernel_size=(1, 5), strides=(1, 1), padding='same', activation='relu')(mp1)
    mp2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(conv2)

    conv3 = Conv2D(filters=filters[2], kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(mp2)
    mp3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(conv3)

    conv4 = Conv2D(filters=1, kernel_size=(1, int(input_shape[1]/8)), strides=(1, 1), padding='valid', activation='relu')(mp3)
    conv4 = Reshape((int(input_shape[1]/8), 1, 1))(conv4)
    conv4 = Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='softmax')(conv4)
    outputs = Flatten()(conv4)
    #outputs = Dense(10, activation='sigmoid')(outputs)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(0.0005), loss=categorical_crossentropy, metrics=['accuracy'])

    return model


def build_dense_small(input_shape: Tuple, num_classes: int, seed: int=0):
    inputs = Input(shape=input_shape, name='input0')
    flatten = Flatten()(inputs)
    outputs = Dense(units=num_classes, activation='softmax', name='output_0', dtype="float32")(flatten)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(0.001), loss=categorical_crossentropy, metrics=['accuracy'])

    return model


def build_in_out(input_shape: Tuple, num_classes: int, seed: int=0):
    inputs = Input(shape=input_shape, name='input0')
    flatten = Flatten()(inputs)
    #outputs = Dense(units=num_classes, activation='softmax', name='output_0', dtype="float32")(reshape)

    model = Model(inputs=[inputs], outputs=[flatten])
    model.compile(optimizer=Adam(0.001), loss=categorical_crossentropy, metrics=['accuracy'])

    return model