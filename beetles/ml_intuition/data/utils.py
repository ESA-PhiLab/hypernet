"""
All data handling methods.
"""

from typing import Dict, List, Tuple, Union

import os
import mlflow
import numpy as np
import tensorflow as tf

from ml_intuition import enums
from ml_intuition.data.transforms import BaseTransform

SAMPLES_DIM = 0
MEAN_PER_CLASS_ACC = 'mean_per_class_accuracy'


def create_tf_dataset(batch_size: int,
                      dataset: Dict[str, np.ndarray],
                      transforms: List[BaseTransform]) -> Tuple[
    tf.data.Dataset, int]:
    """
    Create and transform datasets that are used in the training, validaton or testing phases.

    :param batch_size: Size of the batch used in either phase,
        it is the size of samples per gradient step.
    :param dataset: Passed dataset as a dictionary of samples and labels.
    :param transforms: List of all transformations. 
    :return: Transformed dataset with its size.
    """
    n_samples = dataset[enums.Dataset.DATA].shape[SAMPLES_DIM]
    for f_transform in transforms:
        dataset[enums.Dataset.DATA], dataset[enums.Dataset.LABELS] = \
            f_transform(dataset[enums.Dataset.DATA],
                        dataset[enums.Dataset.LABELS])
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(dataset[enums.Dataset.DATA], dtype=tf.float32),
         tf.convert_to_tensor(dataset[enums.Dataset.LABELS], dtype=tf.uint8)))
    return dataset.batch(batch_size=batch_size, drop_remainder=False) \
               .repeat().prefetch(tf.contrib.data.AUTOTUNE), n_samples


def shuffle_arrays_together(arrays: List[np.ndarray], seed: int = 0):
    """
    Shuffle arbitrary number of arrays together, in-place

    :param arrays: List of np.ndarrays to be shuffled
    :param seed: seed for the random state, defaults to 0
    :raises AssertionError: When provided arrays have different sizes along 
                            first dimension
    """
    assert all(len(array) == len(arrays[0]) for array in arrays)
    for array in arrays:
        random_state = np.random.RandomState(seed)
        random_state.shuffle(array)


def freeze_session(session: tf.Session,
                   keep_var_names: List[str] = None,
                   output_names: List[str] = None,
                   clear_devices: bool = True) -> tf.GraphDef:
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    :param session: The TensorFlow session to be frozen.
    :param keep_var_names: A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    :param output_names: Names of the relevant graph outputs.
    :param clear_devices: Remove the device directives from the graph for better
                          portability.
    :return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name for v in tf.global_variables()).difference(
                keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
    return frozen_graph


def build_data_dict(train_x, train_y, val_x, val_y, test_x, test_y) -> Dict:
    """
    Build data dictionary with following structure:
    'train':
        'data': np.ndarray
        'labels': np.ndarray
    'val':
        'data': np.ndarray
        'labels': np.ndarray
    'test':
        'data': np.ndarray
        'labels' np.ndarray
    'min': float
    'max': float

    :param train_x: Train set
    :param train_y: Train labels
    :param val_x: Validation set
    :param val_y: Validation labels
    :param test_x: Test set
    :param test_y: Test labels
    :return: Dictionary containing train, validation and test subsets.
    """
    data_dict = {}
    train_min, train_max = np.amin(train_x), np.amax(train_x)
    data_dict[enums.DataStats.MIN] = train_min
    data_dict[enums.DataStats.MAX] = train_max

    data_dict[enums.Dataset.TRAIN] = {}
    data_dict[enums.Dataset.TRAIN][enums.Dataset.DATA] = train_x
    data_dict[enums.Dataset.TRAIN][enums.Dataset.LABELS] = train_y

    data_dict[enums.Dataset.VAL] = {}
    data_dict[enums.Dataset.VAL][enums.Dataset.DATA] = val_x
    data_dict[enums.Dataset.VAL][enums.Dataset.LABELS] = val_y

    data_dict[enums.Dataset.TEST] = {}
    data_dict[enums.Dataset.TEST][enums.Dataset.DATA] = test_x
    data_dict[enums.Dataset.TEST][enums.Dataset.LABELS] = test_y
    return data_dict


def merge_datasets(dataset: List[Dict]):
    """
    Merge datasets stored in a list by keys
    :param dataset: List of dict datasets
    :return: One dict with merged keys
    """
    merged_dataset = {enums.Dataset.TRAIN: {}, enums.Dataset.VAL: {}}
    merged_dataset[enums.Dataset.TRAIN][enums.Dataset.DATA] = np.concatenate(
        [dataset[enums.Dataset.TRAIN][enums.Dataset.DATA] for dataset in
         dataset], axis=0)
    merged_dataset[enums.Dataset.TRAIN][enums.Dataset.LABELS] = np.concatenate(
        [dataset[enums.Dataset.TRAIN][enums.Dataset.LABELS] for dataset in
         dataset], axis=0)
    merged_dataset[enums.Dataset.VAL][enums.Dataset.DATA] = np.concatenate(
        [dataset[enums.Dataset.VAL][enums.Dataset.DATA] for dataset in
         dataset], axis=0)
    merged_dataset[enums.Dataset.VAL][enums.Dataset.LABELS] = np.concatenate(
        [dataset[enums.Dataset.VAL][enums.Dataset.LABELS] for dataset in
         dataset], axis=0)
    merged_dataset[enums.DataStats.MIN] = np.amin(
        merged_dataset[enums.Dataset.TRAIN][enums.Dataset.DATA])
    merged_dataset[enums.DataStats.MAX] = np.amax(
        merged_dataset[enums.Dataset.TRAIN][enums.Dataset.DATA])
    return merged_dataset


def restructure_per_class_accuracy(metrics: Dict[str, List[float]]) -> Dict[
    str, List[float]]:
    """
    Restructure mean accuracy values of each class under
    'mean_per_class_accuracy' key, to where each class' accuracy value lays
    under it's specific key
    :param metrics: Dictionary with metric names and corresponding values
    :return: Dictionary with modified per class accuracy
    """
    if MEAN_PER_CLASS_ACC in metrics.keys():
        per_class_acc = {'Class_' + str(i):
                             [item] for i, item in
                         enumerate(*metrics[MEAN_PER_CLASS_ACC])}
        metrics.update(per_class_acc)
        del metrics[MEAN_PER_CLASS_ACC]
    return metrics


def predict_with_graph_in_batches(session: tf.Session, input_node: str,
                                  output_node: str, data: np.ndarray,
                                  batch_size: int = 16384):
    batches = np.array_split(data, len(data) // batch_size)
    outputs = []
    for batch in batches:
        prediction = session.run(output_node, feed_dict={input_node: batch})
        prediction = session.run(tf.argmax(prediction, axis=-1))
        outputs.append(prediction)
    return np.concatenate(outputs, axis=0)


def predict_with_model_in_batches(model: tf.keras.Model,
                                  data: np.ndarray,
                                  batch_size: int = 1024):
    batches = np.array_split(data, len(data) // batch_size)
    outputs = []
    for batch in batches:
        prediction = model.predict(batch, batch_size=batch_size)
        prediction = np.argmax(prediction, axis=-1)
        outputs.append(prediction)
    return np.concatenate(outputs, axis=0)


def list_to_string(list_to_convert: List) -> str:
    """
    Convert provided list to comma separated string
    :param list_to_convert: List to convert
    :return: Comma separated string with values of the provided list
    """
    return ",".join(map(str, list_to_convert))


def get_mlflow_artifacts_path(artifacts_storage_path: str) -> str:
    """
    Find full local artifacts storage path relative artifacts storage path
    :param artifacts_storage_path: Relative artifacts storage path
    :return: Full local path to artifacts
    """
    filter_string = 'parameters.artifacts_storage = \'{}\''.format(artifacts_storage_path)
    result = mlflow.search_runs(filter_string=filter_string)['artifact_uri'][0]
    return os.path.join(result, artifacts_storage_path)


def parse_train_size(train_size: List) -> Union[float, int, List[int]]:
    """
    If single element list provided, convert to int or float based on provided
    value
    :param train_size: Train size as list
    :return: Converted type
    """
    if type(train_size) is not list:
        return train_size
    if len(train_size) == 1:
        train_size = float(train_size[0])
        if 0.0 <= train_size <= 1:
            return float(train_size)
        else:
            return int(train_size)
    else:
        return list(map(int, train_size))


def get_label_indices_per_class(labels, return_uniques: bool = True):
    """
    Extract indices of each class
    :param labels: Data labels
    :param return_uniques: Whether to return unique labels contained in
        labels arg
    :return: List with lists of label indices of consecutive labels
    """
    unique_labels = np.unique(labels)
    label_indices = [np.where(labels == label)[0] for label in unique_labels]
    if return_uniques:
        return label_indices, unique_labels
    else:
        return label_indices
