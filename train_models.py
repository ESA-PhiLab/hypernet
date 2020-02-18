import os
import typing

import clize
import tensorflow as tf

import utils


@utils.check_types(str, str, int, int, int, bool, int, int, int)
def train(model_path: str, data_path: str, batch_size: int,
          epochs: int, verbose: int, shuffle: bool, patience: int,
          sample_size: int, n_classes: int):
    """
    Function for training tensorflow models given datasets.

    :param model_path: Path to the model.
    :param data_path: Path to the input data. Frist dimension of the
        dataset should be the number of samples.
    :param batch_size: Size of the batch used in training phase,
        it is the size of samples per gradient step.
    :param epochs: Number of epochs for model to train.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param shuffle: Boolean indicating whether to shuffle dataset
        before each epoch.
    :param patience: Number of epochs without improvement in order to
        stop the training phase.
    :param sample_size: Size of the input sample.
    :param n_classes: Number of classes.
    """
    train_dict, val_dict = utils.load_data(data_path,
                                           utils.Dataset.TRAIN,
                                           utils.Dataset.VAL)

    N_TRAIN = train_dict[utils.Dataset.DATA].shape[utils.Dataset.SAMPLES_DIM]
    N_VAL = val_dict[utils.Dataset.DATA].shape[utils.Dataset.SAMPLES_DIM]

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_dict[utils.Dataset.DATA], train_dict[utils.Dataset.LABELS]))
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_dict[utils.Dataset.DATA], val_dict[utils.Dataset.LABELS]))

    @utils.check_types(tf.Tensor, tf.Tensor)
    def preprocess(sample: tf.Tensor,
                   label: tf.Tensor) -> typing.List[tf.Tensor]:
        return [tf.reshape(tf.cast(sample, tf.float32), (sample_size, 1)),
                tf.one_hot(tf.cast(label, tf.uint8), (n_classes))]

    train_dataset = train_dataset.map(preprocess)\
        .batch(batch_size=batch_size, drop_remainder=False)\
        .repeat().prefetch(tf.contrib.data.AUTOTUNE)

    val_dataset = val_dataset.map(preprocess)\
        .batch(batch_size=batch_size, drop_remainder=False)\
        .repeat().prefetch(tf.contrib.data.AUTOTUNE)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience)

    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile('adam', 'categorical_crossentropy', metrics=['acc'])
    model.summary()

    artifacts = model.fit(x=train_dataset.make_one_shot_iterator(),
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=val_dataset.make_one_shot_iterator(),
                          callbacks=[callback],
                          steps_per_epoch=N_TRAIN // batch_size,
                          validation_steps=N_VAL // batch_size)

    model.save(filepath=os.path.join(os.path.dirname(model_path),
                                     utils.Model.TRAINED_MODEL))
    print(artifacts)


if __name__ == '__main__':
    clize.run(train)
