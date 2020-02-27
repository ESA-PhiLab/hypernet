import inspect
import typing

import aenum
import h5py
import numpy as np
import tensorflow as tf


class Dataset(aenum.Constant):
    SAMPLES_DIM = 0

    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

    DATA = 'data'
    LABELS = 'labels'


class Model(aenum.Constant):
    TRAINED_MODEL = 'trained_model'


def check_types(*types):
    def function_wrapper(function):
        assert len(types) == len(inspect.signature(function).parameters), \
            'Number of arguments must match the number of possible types.'

        def validate_types(*args, **kwargs):
            for (obj, type_) in zip(args, types):
                assert isinstance(obj, type_), \
                    'Object {0} does not match {1} type.'.format(obj, type_)
            # If all objects are consistent return function:
            return function(*args, **kwargs)
        return validate_types
    return function_wrapper


@check_types(str, int, int, int, str, list)
def _extract_trainable_datasets(data_path: str,
                                batch_size: int,
                                sample_size: int,
                                n_classes: int,
                                dataset_key: str,
                                transforms: list) -> tuple:
    """
    Create datasets that are used in the training and validation phases.

    :param data_path: Path to the input data. Frist dimension of the
        dataset should be the number of samples.
    :param batch_size: Size of the batch used in training phase,
        it is the size of samples per gradient step.
    :param sample_size: Size of the input sample.
    :param n_classes: Number of classes in the dataset.
    :param dataset_key: Key which specifies which dataset to load.
    :param transforms: List of all transformations. 
    """
    dataset = load_data(data_path, dataset_key)
    N_SAMPLES = dataset[Dataset.DATA].shape[Dataset.SAMPLES_DIM]
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataset[Dataset.DATA], dataset[Dataset.LABELS]))
    for transform in transforms:
        dataset = dataset.map(transform)
    return dataset.batch(batch_size=batch_size, drop_remainder=False)\
        .repeat()\
        .prefetch(tf.contrib.data.AUTOTUNE), N_SAMPLES


@check_types(str, str)
def load_data(data_path: str, dataset_key: str) -> dict:
    """
    Function for loading datasets as list of dictionaries.

    :param data_path: Path to the dataset.
    :param keys: Keys for each dataset.
    """
    raw_data = h5py.File(data_path, 'r')
    dataset = {
        Dataset.DATA: np.asarray(raw_data[dataset_key][Dataset.DATA]),
        Dataset.LABELS: np.asarray(
            raw_data[dataset_key][Dataset.LABELS])
    }
    return dataset


def shuffle_arrays_together(arrays: typing.List[np.ndarray], seed: int = 0):
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
                         train_size: typing.Union[int, float] = 0.8,
                         val_size: float = 0.1,
                         balanced: bool = True,
                         background_label: int = 0) -> typing.Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into train, val and test sets. The size of the training set 
    is set by the train_size parameter. All the remaining samples will be
    treated as a test set

    :param data: Data with the [SAMPLES, ...] dimensions
    :param labels: Vector with corresponding labels
    :param train_size: If float, should be between 0.0 and 1.0,
                        if balanced = True, it represents percentage of each
                        class to be extracted,
                 If float and balanced = False, it represents percentage of the
                    whole dataset to be extracted with samples drawn randomly,
                    regardless of their class.
                 If int and balanced = True, it represents number of samples
                    to be drawn from each class.
                 If int and balanced = False, it represents overall number of
                    samples to be drawn regardless of their class, randomly.
                 Defaults to 0.8
    :param val_size: Should be between 0.0 and 1.0. Represents the percentage of
                     each class from the training set to be extracted as a
                     validation set, defaults to 0.1
    :param balanced: Indicated whether the extracted training set should be
                     balanced, defaults to True
    :param background_label: Label indicating the background in the ground truth
    :return: Three tuples: (train_x, train_y), (val_x, val_y), (test_x, test_y)
    :raises TypeError: When wrong type is passed as train_size
    """
    data = data[labels != background_label]
    labels = labels[labels != background_label]
    labels = normalize_labels(labels)
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
    :param size: If float, should be between 0.0 and 1.0, if balanced = True, it
                    represents percentage of each class to be extracted,
                 If float and balanced = False, it represents percentage of the
                    whole dataset to be extracted with samples drawn randomly,
                    regardless of their class.
                 If int and balanced = True, it represents number of samples
                    to be drawn from each class.
                 If int and balanced = False, it represents overall number of
                    samples to be drawn regardless of their class, randomly.
                 Defaults to 0.8
    :param balanced: Indicated whether the extracted training set should be
                     balanced, defaults to True
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


def normalize_labels(labels: np.ndarray) -> np.ndarray:
    """
    Normalize labels so that they always start from 0
    :param labels: labels to normalize
    :return: Normalized labels
    """
    min_label = np.amin(labels)
    return labels - min_label


def reshape_to_1d_samples(data: np.ndarray,
                          labels: np.ndarray,
                          channels_idx: int = 0) -> typing.Tuple[
        np.ndarray, np.ndarray]:
    """
    Reshape the data and labels from [CHANNELS, HEIGHT, WIDTH] to [PIXEL,
    CHANNELS],
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
