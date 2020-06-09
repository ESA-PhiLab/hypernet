"""
All data handling methods.
"""

from typing import Dict, List, Tuple, Union

import os
import json
import yaml
import mlflow
import numpy as np
import tensorflow as tf

from ml_intuition import enums
from ml_intuition.data.transforms import BaseTransform

SAMPLES_DIM = 0
MEAN_PER_CLASS_ACC = 'mean_per_class_accuracy'
LOGGING_EXCLUDED_PARAMS = ['run_name', 'experiment_name', 'use_mlflow', 'verbose']


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


def train_val_test_split(data: np.ndarray, labels: np.ndarray,
                         train_size: Union[List, float, int] = 0.8,
                         val_size: float = 0.1,
                         stratified: bool = True,
                         seed: int = 0) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into train, val and test sets. The size of the training set 
    is set by the train_size parameter. All the remaining samples will be
    treated as a test set

    :param data: Data with the [SAMPLES, ...] dimensions
    :param labels: Vector with corresponding labels
    :param train_size: If float, should be between 0.0 and 1.0,
                        if stratified = True, it represents percentage of each
                        class to be extracted,
                 If float and stratified = False, it represents percentage of the
                    whole dataset to be extracted with samples drawn randomly,
                    regardless of their class.
                 If int and stratified = True, it represents number of samples
                    to be drawn from each class.
                 If int and stratified = False, it represents overall number of
                    samples to be drawn regardless of their class, randomly.
                 Defaults to 0.8
    :param val_size: Should be between 0.0 and 1.0. Represents the percentage of
                     each class from the training set to be extracted as a
                     validation set, defaults to 0.1
    :param stratified: Indicated whether the extracted training set should be
                     stratified, defaults to True
    :param seed: Seed used for data shuffling
    :return: train_x, train_y, val_x, val_y, test_x, test_y
    :raises AssertionError: When wrong type is passed as train_size
    """
    shuffle_arrays_together([data, labels], seed=seed)
    train_indices = _get_set_indices(labels, train_size, stratified)
    val_indices = _get_set_indices(labels[train_indices], val_size)
    val_indices = train_indices[val_indices]
    test_indices = np.setdiff1d(np.arange(len(data)), train_indices)
    train_indices = np.setdiff1d(train_indices, val_indices)
    return data[train_indices], labels[train_indices], data[val_indices], \
           labels[val_indices], data[test_indices], labels[test_indices]


def _get_set_indices(labels: np.ndarray, size: Union[List, float, int] = 0.8,
                     stratified: bool = True) -> np.ndarray:
    """
    Extract indices of a subset of specified data according to size and
    stratified parameters.

    :param labels: Vector with corresponding labels
    :param size: If float, should be between 0.0 and 1.0, if stratified = True, it
                    represents percentage of each class to be extracted,
                 If float and stratified = False, it represents percentage of the
                    whole dataset to be extracted with samples drawn randomly,
                    regardless of their class.
                 If int and stratified = True, it represents number of samples
                    to be drawn from each class.
                 If int and stratified = False, it represents overall number of
                    samples to be drawn regardless of their class, randomly.
                 Defaults to 0.8
    :param stratified: Indicated whether the extracted training set should be
                     stratified, defaults to True
    :return: Indexes of the train set
    :raises TypeError: When wrong type is passed as size
    """
    unique_labels = np.unique(labels)
    label_indices = [np.where(labels == label)[0] for label in unique_labels]
    if isinstance(size, list) and len(size) == 1:
        size = float(size[0])
    if isinstance(size, (float, int)):
        assert size > 0, "Size argument must be greater than zero"
        if 0.0 < size < 1.0 and stratified is True:  # additional condition isinstance
            for idx in range(len(unique_labels)):
                samples_per_label = int(len(label_indices[idx]) * size)
                label_indices[idx] = label_indices[idx][:samples_per_label]
            train_indices = np.concatenate(label_indices, axis=0)
        elif 0.0 < size < 1.0 and stratified is False:
            train_indices = np.arange(int(len(labels) * size))
        elif size >= 1 and stratified is True:
            for label in range(len(unique_labels)):
                label_indices[label] = label_indices[label][:int(size)]
            train_indices = np.concatenate(label_indices, axis=0)
        elif size >= 1 and stratified is False:
            train_indices = np.arange(size, dtype=int)
    elif isinstance(size, list):
        size = list(map(float, size))
        for n_samples, label in zip(size, range(len(unique_labels))):
            label_indices[label] = label_indices[label][:int(n_samples)]
        train_indices = np.concatenate(label_indices, axis=0)
    return train_indices


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


def apply_transformations(data: Dict,
                          transformations: List[BaseTransform]) -> Dict:
    """
    Apply each transformation from provided list
    :param data: Dictionary with 'data' and 'labels' keys holding np.ndarrays
    :param transformations: List of transformations
    :return: Transformed data, in the same format as input
    """
    for transformation in transformations:
        data[enums.Dataset.DATA], data[enums.Dataset.LABELS] = transformation(
            data[enums.Dataset.DATA], data[enums.Dataset.LABELS])
    return data


def list_to_string(list_to_convert: List):
    """
    Convert provided list to comma separated string
    :param list_to_convert: List to convert
    :return: Comma separated string with values of the provided list
    """
    return ",".join(list_to_convert)


def log_dict(dict_as_string: str):
    try:
        to_log = json.loads(dict_as_string)
    except Exception:
        to_log = yaml.load(dict_as_string)
    mlflow.log_params(to_log)


def log_params_to_mlflow(args: Dict) -> None:
    """
    Log provided arguments as dictionary to mlflow.
    :param args: Arguments to log
    """
    for arg in args.keys():
        if arg not in LOGGING_EXCLUDED_PARAMS and args[arg] is not None:
            if type(args[arg]) is list:
                args[arg] = list_to_string(args[arg])
                if args[arg] == "":
                    continue
            elif arg == 'noise_params':
                log_dict(args[arg])
                continue
            mlflow.log_param(arg, args[arg])


def get_mlflow_artifacts_path(artifacts_storage_path: str) -> str:
    """
    Find full local artifacts storage path relative artifacts storage path
    :param artifacts_storage_path: Relative artifacts storage path
    :return: Full local path to artifacts
    """
    filter_string = 'parameters.artifacts_storage = \'{}\''.format(artifacts_storage_path)
    result = mlflow.search_runs(filter_string=filter_string)['artifact_uri'][0]
    return os.path.join(result, artifacts_storage_path)
