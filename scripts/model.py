import tensorflow as tf


def build_1d_model(input_shape: tuple, filters: int,
                   kernel_size: int, classes_count: int, blocks: int = 1):
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)

    def add_block(model):
        model.add(tf.keras.layers.Conv1D(filters, kernel_size,
                                         padding="valid", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        return model

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters, kernel_size,
                                     input_shape=input_shape,
                                     padding="valid", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    if blocks > 1:
        for block in range(blocks - 1):
            model = add_block(model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=200, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=classes_count, activation='softmax'))
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')
    return model


if __name__ == '__main__':
    build_1d_model((103, 1), 4, 3, 9, 1).save('model')
