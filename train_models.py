import tensorflow as tf
import utils
import clize


@utils.check_types(str)
def load_data(data_path: str):
    pass


@utils.check_types(str, str, int, int, int, bool, int)
def train(model_path: str, data_path: str, batch_size: int,
          epochs: int, verbose: int, shuffle: bool, patience: int):
    """
    Function for training tensorflow models given datasets.

    :param model_path: Path to the model.
    :param data_path: Path to the input data.
    :param batch_size: Size of the batch used in training phase,
        it is the size of samples per gradient step.
    :param epochs: Number of epochs for model to train.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param shuffle: Boolean indicating whether to shuffle dataset
        before each epoch.
    :param patience: Number of epochs without improvement in order to
        stop the training phase.
    """
    data = load_data(data_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience)

    model.compile('adam', 'categorical_crossentropy', metrics=['acc'])
    model.summary()

    model.fit(x=data[utils.Dataset.TRAIN],
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              shuffle=shuffle,
              validation_data=data[utils.data[utils.Dataset.VAL]],
              callbacks=callback)
    print('Passed...')


if __name__ == '__main__':
    # Mocked args:
    clize.run(train)
