import sys

import tensorflow as tf


def model_1d(kernel_size: int,
             n_kernels: int,
             n_layers: int,
             input_size: tuple,
             n_classes: int,
             lr: float = 0.001):
    def add_layer(model):
        model.add(tf.keras.layers.Conv1D(n_kernels, kernel_size,  # (kernel_size, 1)
                                         padding="valid",
                                         activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        return model

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(n_kernels, kernel_size,
                                     input_shape=(input_size, 1),
                                     padding="valid",
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    if n_layers > 1:
        for _ in range(n_layers - 1):
            model = add_layer(model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=200, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))

    return model


def get_model(model_key: str, kernel_size: int, n_kernels: int,
              n_layers: int, input_size: int, n_classes: int,
              lr: float = 0.001):
    all_ = {
        str(f): eval(f) for f in dir(sys.modules[__name__])
    }
    return all_[model_key](kernel_size=kernel_size,
                           n_kernels=n_kernels, n_layers=n_layers,
                           input_size=input_size, n_classes=n_classes, lr=lr)
