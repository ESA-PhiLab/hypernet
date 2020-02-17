import os

import clize
import tensorflow as tf

import utils


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
    train_data, val_data = utils.load_data(data_path,
                                           utils.Dataset.TRAIN,
                                           utils.Dataset.VAL)

    model = tf.keras.models.load_model(model_path, compile=False)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience)

    model.compile('adam', 'categorical_crossentropy', metrics=['acc'])
    print(model.summary())

    model.fit(x=train_data,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              shuffle=shuffle,
              validation_data=val_data,
              callbacks=callback)

    model.save(filepath=os.path.join(os.path.dirname(model_path),
                                     utils.Model.TRAINED_MODEL))


if __name__ == '__main__':
    clize.run(train)
