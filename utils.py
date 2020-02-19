import aenum
from typing import Tuple, List, Union

import tensorflow as tf
import numpy as np


class Dataset(aenum.Constant):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    DATA = 'data'
    LABELS = 'labels'


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
                         train_size: Union[int, float] = 0.8,
                         val_size: float = 0.1,
                         balanced: bool = True,
                         background_label: int = 0) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into train, val and test sets. The size of the training set 
    is set by the train_size paramter. All the remaining samples will be treated
    as a test set

    :param data: Data with the [SAMPLES, ...] dimensions
    :param labels: Vector with corresponding labels
    :param train_size: If float, should be between 0.0 and 1.0, if balanced = True, it represents percentage of each class to be extracted, 
                       If float and balanced = False, it represents percetange of the whole dataset to be extracted with samples drawn randomly, 
                            regardlesss of their class. 
                       If int and balanced = True, it represents number of samples to be drawn from each class. 
                       If int and balanced = False, it represents overall number of samples to be drawn regardless of their class, randomly. 
                       Defaults to 0.8
    :param val_size: Should be between 0.0 and 1.0. Represents the percentage of each class from the training set 
                     to be extracted as a validation set, defaults to 0.1
    :param balanced: Indicated whether the extracted training set should be balanced, defaults to True
    :return: Three tuples: (train_x, train_y), (val_x, val_y), (test_x, test_y)
    :raises TypeError: When wrong type is passed as train_size
    """
    data = data[labels != background_label]
    labels = labels[labels != background_label]
    shuffle_arrays_together([data, labels])
    train_indices = _get_set_indices(labels, train_size, balanced)
    train_x = data[train_indices]
    train_y = labels[train_indices]
    val_indices = _get_set_indices(train_y, val_size)
    val_x = train_x[val_indices]
    val_y = train_y[val_indices]
    train_x = np.delete(train_x, val_indices, axis=0)
    train_y = np.delete(train_y, val_indices)
    data = np.delete(data, train_indices, axis=0)
    labels = np.delete(labels, train_indices)
    return train_x, train_y, val_x, val_y, data, labels


def _get_set_indices(labels: np.ndarray, size: float = 0.8,
                     balanced: bool = True) -> np.ndarray:
    """
    Extract indices of a subset of specified data according to size and
    balanced parameters.

    :param labels: Vector with corresponding labels
    :param size: If float, should be between 0.0 and 1.0, if balanced = True, it represents percentage of each class to be extracted, 
                       If float and balanced = False, it represents percetange of the whole dataset to be extracted with samples drawn randomly, 
                            regardless of their class.
                       If int and balanced = True, it represents number of samples to be drawn from each class. 
                       If int and balanced = False, it represents overall number of samples to be drawn regardless of their class, randomly. 
                       Defaults to 0.8
    :param balanced: Indicated whether the extracted training set should be balanced, defaults to True
    :return: Indexes of the train set
    :raises TypeError: When wrong type is passed as size
    """
    unique_labels = np.unique(labels)
    label_indices = [np.where(labels == label)[0] for label in unique_labels]
    if 0.0 < size < 1.0 and balanced is True:
        for idx in range(len(unique_labels)):
            samples_per_label = int(len(label_indices[idx]) * size)
            label_indices[idx] = label_indices[idx][:samples_per_label]
        train_indices = np.concatenate(label_indices, axis=0)
    elif 0.0 < size < 1.0 and balanced is False:
        train_indices = np.arange(int(len(labels) * size))
    elif size >= 1 and balanced is True:
        for label in range(len(unique_labels)):
            label_indices[label] = label_indices[label][:size]
        train_indices = np.concatenate(label_indices, axis=0)
    elif size >= 1 and balanced is False:
        train_indices = np.arange(size)
    else:
        raise TypeError("Wrong type of size argument passed")
    return train_indices


def reshape_to_1d_samples(data: np.ndarray,
                          labels: np.ndarray,
                          channels_idx: int = 0) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Reshape the data and labels from [CHANNELS, HEIGHT, WIDTH] to [PIXEL, CHANNELS],
    so it fits the 1D Conv models
    :param data: Data to reshape.
    :param labels: Corresponding labels.
    :param channels_idx: Index at which the channels are located in the
                         provided data file
    :return: Reshape data and labels
    :rtype: tuple with reshaped data and labels
    """
    data = data.reshape(data.shape[channels_idx], -1)
    data = np.moveaxis(data, -1, 0)
    labels = labels.reshape(-1)
    return data, labels


def freeze_session(session, keep_var_names=None, output_names=None,
                   clear_devices=True):
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
