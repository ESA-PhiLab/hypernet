import clize
import utils
import tensorflow as tf


@utils.check_types(str, str, int, int)
def evaluate(model_path: str, data_path: str,
             batch_size: int, verbose: int):
    """
    Function for evaluating the trained model.

    :param model_path: Path to the model.
    :param data_path: Path to the input data.
    :param batch_size: Size of the batch used in evaluation step.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    """
    test_data = utils.load_data(data_path, utils.Dataset.TEST)
    model = tf.keras.models.load_model(model_path, compile=False)
    model.evaluate(x=test_data, batch_size=batch_size, verbose=verbose)


if __name__ == '__main__':
    clize.run(evaluate)
