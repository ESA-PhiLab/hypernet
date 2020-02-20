import clize
import tensorflow as tf

import transform
import utils


@utils.check_types(str, str, int, int, int, int)
def evaluate(model_path: str, data_path: str,
             batch_size: int, verbose: int,
             sample_size: int, n_classes: int):
    """
    Function for evaluating the trained model.

    :param model_path: Path to the model.
    :param data_path: Path to the input data.
    :param batch_size: Size of the batch used in evaluation step.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param sample_size: Size of the input sample.
    :param n_classes: Number of classes.
    """
    test_dict = utils.load_data(data_path, utils.Dataset.TEST)[0]

    N_TEST = test_dict[utils.Dataset.DATA].shape[utils.Dataset.SAMPLES_DIM]

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_dict[utils.Dataset.DATA], test_dict[utils.Dataset.LABELS]))

    transormations = transform.Transform1D(sample_size=sample_size,
                                           n_classes=n_classes)

    test_dataset = test_dataset.map(transormations)\
        .batch(batch_size=batch_size, drop_remainder=False)\
        .repeat().prefetch(tf.contrib.data.AUTOTUNE)

    model = tf.keras.models.load_model(model_path, compile=True)
    artifacts = model.evaluate(x=test_dataset.make_one_shot_iterator(),
                               verbose=verbose,
                               steps=N_TEST//batch_size)
    print(artifacts)


if __name__ == '__main__':
    clize.run(evaluate)
