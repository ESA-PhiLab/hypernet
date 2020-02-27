import clize
import tensorflow as tf

from ml_intuition.data import transform, utils


@utils.check_types(str, str, int, int, int, int)
def evaluate(*,
             model_path: str,
             data_path: str,
             batch_size: int,
             verbose: int,
             sample_size: int,
             n_classes: int):
    """
    Function for evaluating the trained model.

    :param model_path: Path to the model.
    :param data_path: Path to the input data.
    :param batch_size: Size of the batch used in evaluation step.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param sample_size: Size of the input sample.
    :param n_classes: Number of classes.
    """
    test_dataset, N_TEST =\
        utils._extract_trainable_datasets(data_path,
                                          batch_size,
                                          sample_size,
                                          n_classes,
                                          utils.Dataset.TEST,
                                          [transform.Transform1D(sample_size,
                                                                 n_classes)])
    model = tf.keras.models.load_model(model_path, compile=True)
    artifacts = model.evaluate(x=test_dataset.make_one_shot_iterator(),
                               verbose=verbose,
                               steps=N_TEST//batch_size)
    print(artifacts)


if __name__ == '__main__':
    clize.run(evaluate)
